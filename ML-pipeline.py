import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import sqlite3
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ PHYSICS-INFORMED FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════════════

class PhysicsInformedFeatures:
    """
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │ Physics-Informed Feature Extractor                                                  │
    │ ─────────────────────────────────────────────────────────────────────────────────── │
    │ • Computes traditional seismological features that help with phase identification   │
    │ • STA/LTA ratios at multiple time scales                                            │
    │ • Frequency domain features                                                         │
    │ • Envelope and instantaneous phase                                                  │
    │ • These features provide domain knowledge to guide neural network learning          │
    └─────────────────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        
        # STA/LTA parameters for different time scales
        self.sta_lta_configs = [
            {'sta': 0.5, 'lta': 10.0},   # Fast detection
            {'sta': 1.0, 'lta': 20.0},   # Medium scale
            {'sta': 2.0, 'lta': 30.0},   # Slow, stable detection
        ]
        
        # Frequency bands for spectral analysis
        self.freq_bands = [
            {'name': 'low', 'freqmin': 1.0, 'freqmax': 5.0},      # Low frequency
            {'name': 'mid', 'freqmin': 5.0, 'freqmax': 15.0},     # Mid frequency
            {'name': 'high', 'freqmin': 15.0, 'freqmax': 45.0},   # High frequency
        ]
    
    def compute_sta_lta_features(self, waveform):
        """Compute STA/LTA ratios at multiple time scales"""
        from obspy.signal.trigger import classic_sta_lta
        
        features = []
        
        for config in self.sta_lta_configs:
            sta_samples = int(config['sta'] * self.sampling_rate)
            lta_samples = int(config['lta'] * self.sampling_rate)
            
            # Ensure we have enough samples
            if len(waveform) < lta_samples:
                print(f"Warning: Waveform too short ({len(waveform)}) for LTA ({lta_samples})")
                # Create zeros as fallback
                features.extend([np.zeros_like(waveform), np.zeros_like(waveform)])
                continue
            
            try:
                # Compute classic STA/LTA
                sta_lta = classic_sta_lta(waveform, sta_samples, lta_samples)
                features.append(sta_lta)
                
                # Also compute log of STA/LTA for better dynamic range
                log_sta_lta = np.log10(np.maximum(sta_lta, 1e-10))
                features.append(log_sta_lta)
            except Exception as e:
                print(f"Warning: STA/LTA computation failed: {e}")
                # Add zeros as fallback
                features.extend([np.zeros_like(waveform), np.zeros_like(waveform)])
        
        return np.array(features)
    
    def compute_envelope_features(self, waveform):
        """Compute envelope and instantaneous features"""
        try:
            # Analytic signal for envelope and instantaneous phase
            analytic_signal = signal.hilbert(waveform)
            envelope = np.abs(analytic_signal)
            instantaneous_phase = np.angle(analytic_signal)
            
            # Envelope derivative (rate of change)
            envelope_derivative = np.gradient(envelope)
            
            # Instantaneous frequency
            instantaneous_freq = np.gradient(np.unwrap(instantaneous_phase)) / (2.0 * np.pi) * self.sampling_rate
            
            return np.array([
                envelope,
                envelope_derivative,
                instantaneous_freq
            ])
        except Exception as e:
            print(f"Warning: Envelope computation failed: {e}")
            # Return zeros as fallback
            return np.array([
                np.zeros_like(waveform),
                np.zeros_like(waveform), 
                np.zeros_like(waveform)
            ])
    
    def compute_spectral_features(self, waveform):
        """Compute frequency domain features"""
        from obspy.signal.filter import bandpass
        
        features = []
        
        for band in self.freq_bands:
            # Bandpass filter
            try:
                filtered = bandpass(waveform, band['freqmin'], band['freqmax'], 
                                  self.sampling_rate, corners=2, zerophase=True)
                
                # Energy in this band
                energy = filtered ** 2
                features.append(energy)
                
                # Envelope of filtered signal
                envelope = np.abs(signal.hilbert(filtered))
                features.append(envelope)
                
            except Exception as e:
                print(f"Warning: Could not compute {band['name']} band features: {e}")
                # Add zeros as fallback
                features.extend([np.zeros_like(waveform), np.zeros_like(waveform)])
        
        return np.array(features)
    
    def compute_all_features(self, waveform):
        """
        Compute all physics-informed features
        
        Args:
            waveform: 1D numpy array of seismic data
            
        Returns:
            features: 2D numpy array of shape (n_features, n_samples)
        """
        # Normalize waveform to prevent numerical issues
        waveform_norm = waveform / (np.std(waveform) + 1e-10)
        
        # Compute different feature types
        sta_lta_features = self.compute_sta_lta_features(waveform_norm)
        envelope_features = self.compute_envelope_features(waveform_norm)
        spectral_features = self.compute_spectral_features(waveform_norm)
        
        # Combine all features
        all_features = np.vstack([
            waveform_norm.reshape(1, -1),  # Original waveform
            sta_lta_features,              # STA/LTA features
            envelope_features,             # Envelope features  
            spectral_features              # Spectral features
        ])
        
        return all_features

class ShiftGradientFeature:
    @staticmethod
    def compute(waveform: np.ndarray, target_length: int) -> np.ndarray:
        # 1) Convert all negative values to positive (absolute value)
        abs_waveform = np.abs(waveform)
        # 2) gradient (current – previous)
        grad = np.diff(abs_waveform, prepend=abs_waveform[0])
        # 3) resize to target_length (use the same pad‐or‐truncate logic you already have)
        if len(grad) > target_length:
            # truncate center
            start = (len(grad) - target_length) // 2
            grad = grad[start:start+target_length]
        elif len(grad) < target_length:
            # pad at end
            grad = np.pad(grad, (0, target_length-len(grad)), 'constant')
        return grad.reshape(1, -1)

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ ENHANCED U-NET ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Enhanced convolution block with batch normalization and dropout"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.1):
        super().__init__()
        
        self.doubleConv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),
            
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout)
        )
        
    def forward(self, x):
        return self.doubleConv(x)

class AttentionBlock(nn.Module):
    """Attention mechanism to focus on important features"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x

class AdaptiveUNet1D(nn.Module):
    """
    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │ Adaptive 1-D U-Net                                                                   │
    │ ──────────────────────────────────────────────────────────────────────────────────── │
    │ • Can work with or without physics-informed features                                 │
    │ • Attention mechanisms to focus on important features                                │
    │ • Batch normalization and dropout for better generalization                          │
    │ • Multi-scale feature extraction through encoder-decoder architecture                │
    │ • Dynamically adapts to different input channel sizes                                │
    └──────────────────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, in_channels=1, out_channels=2, features=[32, 64, 128, 256], 
                 dropout=0.1, use_physics_features=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.use_physics_features = use_physics_features
        
        # ==============================
        # 1️⃣ Downsampling Path (ENCODER)
        # ==============================
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.attentions_down = nn.ModuleList()
        
        current_channels = in_channels
        for feat in features:
            self.downs.append(ConvBlock(current_channels, feat, dropout=dropout))
            self.pools.append(nn.MaxPool1d(2))
            self.attentions_down.append(AttentionBlock(feat))
            current_channels = feat
        
        # ============================================
        # 2️⃣ Bottleneck (connects ENCODER & DECODER)
        # ============================================
        self.bottleneck = ConvBlock(features[-1], features[-1]*2, dropout=dropout)
        self.bottleneck_attention = AttentionBlock(features[-1]*2)
        
        # ==============================
        # 3️⃣ Upsampling path (DECODER)
        # ==============================
        self.ups = nn.ModuleList()
        self.attentions_up = nn.ModuleList()
        
        for feat in reversed(features):
            # Transposed convolution for upsampling
            self.ups.append(nn.ConvTranspose1d(feat*2, feat, kernel_size=2, stride=2))
            # Convolution block for feature fusion
            self.ups.append(ConvBlock(feat*2, feat, dropout=dropout))
            # Attention for refined features
            self.attentions_up.append(AttentionBlock(feat))
        
        # ===========================
        # 4️⃣ Final Classification Layer
        # ===========================
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)
        
        # Physics-informed feature weighting (only if using physics features)
        self.feature_weights = None
        if use_physics_features:
            self.feature_weights = nn.Parameter(torch.ones(in_channels))
        
    def initialize_feature_weights(self, actual_channels):
        """Initialize feature weights based on actual number of input channels"""
        if self.use_physics_features:
            if self.feature_weights is None or self.feature_weights.size(0) != actual_channels:
                self.feature_weights = nn.Parameter(torch.ones(actual_channels))
                self.in_channels = actual_channels
                print(f"Initialized feature weights for {actual_channels} channels")
        
    def forward(self, x):
        # Apply learnable weights to input features (only if using physics features)
        if self.use_physics_features:
            # Initialize feature weights if needed
            if self.feature_weights is None:
                self.initialize_feature_weights(x.size(1))
            
            if x.size(1) == self.feature_weights.size(0):
                weighted_x = x * self.feature_weights.view(1, -1, 1)
            else:
                print(f"Warning: Input channels ({x.size(1)}) != feature weights ({self.feature_weights.size(0)})")
                # Adjust feature weights if mismatch
                self.initialize_feature_weights(x.size(1))
                weighted_x = x * self.feature_weights.view(1, -1, 1)
        else:
            weighted_x = x
        
        skip_connections = []
        
        # ---------------- Encoder ----------------
        for i, (down, pool, attention) in enumerate(zip(self.downs, self.pools, self.attentions_down)):
            weighted_x = down(weighted_x)
            weighted_x = attention(weighted_x)  # Apply attention
            skip_connections.append(weighted_x)
            weighted_x = pool(weighted_x)
        
        # --------------- Bottleneck ---------------
        weighted_x = self.bottleneck(weighted_x)
        weighted_x = self.bottleneck_attention(weighted_x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # ---------------- Decoder ----------------
        for idx in range(0, len(self.ups), 2):
            # Upsampling
            weighted_x = self.ups[idx](weighted_x)
            
            # Get corresponding skip connection
            skip_conn = skip_connections[idx//2]
            
            # Handle size mismatches
            if weighted_x.shape[-1] != skip_conn.shape[-1]:
                weighted_x = F.pad(weighted_x, (0, skip_conn.shape[-1] - weighted_x.shape[-1]))
            
            # Concatenate skip connection
            weighted_x = torch.cat((skip_conn, weighted_x), dim=1)
            
            # Refine features
            weighted_x = self.ups[idx+1](weighted_x)
            
            # Apply attention
            attention_idx = idx // 2
            if attention_idx < len(self.attentions_up):
                weighted_x = self.attentions_up[attention_idx](weighted_x)
        
        # Final classification
        output = self.final_conv(weighted_x)
        return F.softmax(output, dim=1)

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ DATASET LOADING FROM DATABASE
# ═══════════════════════════════════════════════════════════════════════════════════════

class SeismicDatabaseDataset(Dataset):
    """
    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │ Seismic Database Dataset                                                             │
    │ ──────────────────────────────────────────────────────────────────────────────────── │
    │ • Loads waveforms from SQLite database created by data_mine.py                       │
    │ • Optionally computes physics-informed features                                      │
    │ • Generates labels for P-wave detection                                              │
    └──────────────────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, db_path='seismic_data.db', window_size=5, target_length=12300, 
                 use_physics_features=True, use_shift_gradient=True, 
                 train=True, train_split=0.8, random_seed=42):
        self.db_path = db_path
        self.window_size = window_size
        self.target_length = target_length
        self.use_physics_features = use_physics_features
        self.use_shift_gradient = use_shift_gradient
        
        # Initialize physics feature extractor if needed
        if self.use_physics_features:
            self.feature_extractor = PhysicsInformedFeatures(sampling_rate=100)
        
        # Load data from database
        self._load_from_database(train, train_split, random_seed)
        
    def _load_from_database(self, train, train_split, random_seed):
        """Load waveforms from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First, check what tables exist in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Available tables in database: {[t[0] for t in tables]}")
        
        # Get all waveforms
        cursor.execute("""
            SELECT id, earthquake_id, station_code, waveform_data, 
                   sampling_rate, p_pick_time, eq_time, pre_time, post_time
            FROM waveforms
            ORDER BY earthquake_id, station_code
        """)
        
        all_records = cursor.fetchall()
        conn.close()
        
        if not all_records:
            print("Warning: No records found in database!")
            self.data = []
            self.labels = []
            self.metadata = []
            return
        
        # Shuffle and split data
        np.random.seed(random_seed)
        np.random.shuffle(all_records)
        
        split_idx = int(len(all_records) * train_split)
        if train:
            records = all_records[:split_idx]
        else:
            records = all_records[split_idx:]
        
        print(f"Loading {'training' if train else 'validation'} set: {len(records)} waveforms")
        
        # Process waveforms
        self.data = []
        self.labels = []
        self.metadata = []
        
        for record in tqdm(records, desc="Processing waveforms"):
            waveform_id, eq_id, station, waveform_blob, sr, p_pick, eq_time, pre_time, post_time = record
            
            # Deserialize waveform
            waveform = pickle.loads(waveform_blob)
            
            # Process waveform
            result = self._process_waveform(
                waveform, p_pick, eq_time, pre_time, post_time, sr, station
            )
            
            if result is not None:
                self.data.append(result['features'])
                self.labels.append(result['label'])
                self.metadata.append({
                    'waveform_id': waveform_id,
                    'earthquake_id': eq_id,
                    'station': station,
                    'sampling_rate': sr
                })
        
        print(f"Successfully processed {len(self.data)} waveforms")
    
    def _process_waveform(self, waveform, p_pick_time, eq_time, pre_time, post_time, 
                     sampling_rate, station):
        """Process individual waveform"""
        try:
            # Calculate pick sample index
            total_duration = pre_time + post_time
            pick_offset = p_pick_time - (eq_time - pre_time)
            pick_sample = int(pick_offset * sampling_rate)
            
            # Calculate noise characteristics from first 5 samples
            noise_std = np.std(waveform[:min(5, len(waveform))])
            if noise_std == 0 or np.isnan(noise_std):
                noise_std = 1e-6  # Small default value to avoid zero
            
            # Normalize waveform length
            current_length = len(waveform)
            
            if current_length == self.target_length:
                waveform_norm = waveform.copy()
                pick_sample_norm = pick_sample
            elif current_length > self.target_length:
                # Truncate so the pick falls at one-third of the window
                desired_pick_idx = self.target_length // 3
                start_trim = pick_sample - desired_pick_idx
                # Clamp to valid range
                if start_trim < 0:
                    start_trim = 0
                if start_trim + self.target_length > current_length:
                    start_trim = current_length - self.target_length
                end_idx = start_trim + self.target_length
                waveform_norm = waveform[start_trim:end_idx]
                pick_sample_norm = pick_sample - start_trim
            else:
                # Pad with random Gaussian noise based on first 5 samples
                pad_needed = self.target_length - current_length
                
                # Generate random Gaussian noise
                noise_padding = np.random.normal(0, noise_std, pad_needed)
                
                # Pad at the beginning with noise instead of zeros
                waveform_norm = np.concatenate([noise_padding, waveform])
                pick_sample_norm = pick_sample + pad_needed
            
            # Prepare features
            if self.use_physics_features:
                # Extract physics-informed features
                physics_features = self.feature_extractor.compute_all_features(waveform_norm)
                final_features = physics_features
            else:
                # Use only raw waveform
                final_features = waveform_norm.reshape(1, -1)
            
            if self.use_shift_gradient:
                sg_feat = ShiftGradientFeature.compute(waveform_norm, self.target_length)
                final_features = np.vstack([final_features, sg_feat])
            
            # Ensure features have correct length
            if final_features.shape[1] != self.target_length:
                # Resize each feature channel
                resized = []
                for ch in final_features:
                    length = len(ch)
                    if length > self.target_length:
                        excess = length - self.target_length
                        start = excess // 2
                        resized.append(ch[start:start + self.target_length])
                    elif length < self.target_length:
                        pad = self.target_length - length
                        # For feature padding, still use zeros or repeat edge values
                        resized.append(np.pad(ch, (0, pad), mode='constant', constant_values=0))
                    else:
                        resized.append(ch)
                final_features = np.vstack(resized)
            
            # Create labels
            label = np.zeros(self.target_length)
            
            # Window around pick
            window_samples = int(self.window_size * sampling_rate / 6)
            pick_sample_norm = np.clip(pick_sample_norm, 0, self.target_length - 1)
            
            start_idx = max(0, pick_sample_norm - window_samples)
            end_idx = min(self.target_length, pick_sample_norm + window_samples)
            label[start_idx:end_idx] = 1
            
            return {
                'features': final_features.astype(np.float32),
                'label': label.astype(np.int64)
            }
            
        except Exception as e:
            print(f"Error processing waveform from station {station}: {e}")
            return None
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor(self.labels[idx])
        return features, label

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ TRAINING AND EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the adaptive model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Loss function with class weighting (background vs P-wave)
    class_weights = torch.FloatTensor([1.0, 3.0]).to(device)  # Higher weight for P-wave
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with learning rate scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, 
                                                       desc=f'Epoch {epoch+1}/{num_epochs}')):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                val_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_adaptive_model.pth')
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if patience_counter >= 10:
            print("Early stopping triggered")
            break
    
    return train_losses, val_losses

def evaluate_picks(model, val_dataset, threshold=0.5):
    """Evaluate pick accuracy"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    pick_errors = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            features, true_label = val_dataset[i]
            features = features.unsqueeze(0).to(device)
            
            output = model(features)
            prob = output[0, 1, :].cpu().numpy()  # P-wave probability
            
            # Find predicted pick (maximum probability)
            pred_pick_sample = np.argmax(prob)
            
            # Find true pick (center of labeled window)
            true_pick_samples = np.where(true_label == 1)[0]
            if len(true_pick_samples) > 0:
                true_pick_sample = np.mean(true_pick_samples)
                
                # Calculate error in samples
                error_samples = abs(pred_pick_sample - true_pick_sample)
                pick_errors.append(error_samples)
    
    return np.array(pick_errors)

def visualize_features(dataset, sample_idx=0, save_path='features.png', use_physics_features=True):
    """Visualize the features for a sample"""
    if sample_idx >= len(dataset):
        print(f"Sample index {sample_idx} out of range")
        return
    
    features, label = dataset[sample_idx]
    features_np = features.numpy()
    label_np = label.numpy()
    
    # Feature names based on mode
    if use_physics_features:
        feature_names = [
            'Raw Waveform',
            'STA/LTA (0.5/10s)', 'Log STA/LTA (0.5/10s)',
            'STA/LTA (1/20s)', 'Log STA/LTA (1/20s)', 
            'STA/LTA (2/30s)', 'Log STA/LTA (2/30s)',
            'Envelope', 'Envelope Derivative', 'Instantaneous Frequency',
            'Low Freq Energy', 'Low Freq Envelope',
            'Mid Freq Energy', 'Mid Freq Envelope', 
            'High Freq Energy', 'High Freq Envelope'
        ]
        title_suffix = "Physics-Informed Features"
    else:
        feature_names = ['Raw Waveform']
        title_suffix = "Raw Waveform Only"
    
    # Account for shift gradient feature if present
    if features_np.shape[0] > len(feature_names):
        feature_names.append('Shift Gradient')
    
    # Truncate if we have fewer features than expected
    n_features = min(len(feature_names), features_np.shape[0])
    
    # Create time vector
    n_samples = features_np.shape[1]
    time_vector = np.linspace(-3, 120, n_samples)  # Assuming 3s pre, 120s post
    
    # Create subplot grid
    if n_features == 1:
        # Single plot for raw waveform only
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        axes = [ax]
        n_rows, n_cols = 1, 1
    else:
        # Multiple plots for physics features
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        
        # Plot feature
        ax.plot(time_vector, features_np[i], 'b-', linewidth=1, alpha=0.8)
        
        # Highlight P-wave region
        p_wave_mask = label_np == 1
        if np.any(p_wave_mask):
            p_wave_times = time_vector[p_wave_mask]
            ax.axvspan(p_wave_times[0], p_wave_times[-1], alpha=0.3, color='red', 
                      label='P-wave Window')
        
        # Add earthquake time reference
        ax.axvline(0, color='orange', linestyle=':', alpha=0.7, label='Earthquake Time')
        
        ax.set_title(feature_names[i] if i < len(feature_names) else f'Feature {i}')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        
        if i == 0:  # Add legend only to first subplot
            ax.legend()
    
    # Hide empty subplots
    if n_features > 1:
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
    
    plt.suptitle(f'Seismic Features Visualization - {title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Features visualization saved to {save_path}")

def plot_model_predictions(model, dataset, num_examples=5, save_path='model_predictions.png', 
                          use_physics_features=True):
    """Plot model predictions with features"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Adjust subplot layout based on features mode
    if use_physics_features:
        fig, axes = plt.subplots(num_examples, 3, figsize=(18, 4*num_examples))
        plot_cols = 3
    else:
        fig, axes = plt.subplots(num_examples, 2, figsize=(12, 4*num_examples))
        plot_cols = 2
    
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(min(num_examples, len(dataset))):
            features, true_label = dataset[i]
            
            # Get metadata
            station_info = dataset.metadata[i]
            
            # Get model prediction
            features_batch = features.unsqueeze(0).to(device)
            output = model(features_batch)
            prob = output[0, 1, :].cpu().numpy()  # P-wave probability
            
            # Time vector
            sampling_rate = station_info['sampling_rate']
            pre_time = 3  # Default values from data_mine.py
            post_time = 120
            total_samples = features.shape[1]
            time_vector = np.linspace(-pre_time, post_time, total_samples)
            
            # Raw waveform
            ax1 = axes[i, 0]
            raw_waveform = features[0].numpy()  # First channel is raw waveform
            ax1.plot(time_vector, raw_waveform, 'k-', linewidth=0.8, alpha=0.8)
            
            # Add picks
            true_pick_samples = np.where(true_label == 1)[0]
            model_pick_sample = np.argmax(prob)
            
            if len(true_pick_samples) > 0:
                true_pick_time = time_vector[int(np.mean(true_pick_samples))]
                ax1.axvline(true_pick_time, color='red', linestyle='--', linewidth=2, 
                           label=f'True Pick ({true_pick_time:.2f}s)')
            
            model_pick_time = time_vector[model_pick_sample]
            ax1.axvline(model_pick_time, color='blue', linestyle=':', linewidth=1, 
                       label=f'Model Pick ({model_pick_time:.2f}s)')
            ax1.axvline(0, color='orange', linestyle=':', alpha=0.7, label='Earthquake Time')
            
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.set_title(f'Station {station_info["station"]} - Raw Waveform')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Physics features plot (only if using physics features)
            if use_physics_features and plot_cols == 3:
                ax2 = axes[i, 1]
                if features.shape[0] > 1:  # Check if we have STA/LTA features
                    sta_lta = features[1].numpy()  # Second channel should be STA/LTA
                    ax2.plot(time_vector, sta_lta, 'g-', linewidth=1.5, label='STA/LTA (0.5/10s)')
                    ax2.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='Typical Threshold')
                else:
                    ax2.text(0.5, 0.5, 'No STA/LTA features', transform=ax2.transAxes, ha='center')
                
                if len(true_pick_samples) > 0:
                    ax2.axvline(true_pick_time, color='red', linestyle='--', linewidth=2)
                ax2.axvline(model_pick_time, color='blue', linestyle='-', linewidth=2)
                ax2.axvline(0, color='orange', linestyle=':', alpha=0.7)
                
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('STA/LTA Ratio')
                ax2.set_title(f'Station {station_info["station"]} - STA/LTA Feature')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                prob_col = 2
            else:
                prob_col = 1
            
            # Model probability
            ax3 = axes[i, prob_col]
            ax3.plot(time_vector, prob, 'b-', linewidth=2, label='P-wave Probability')
            ax3.fill_between(time_vector, 0, prob, alpha=0.3, color='blue')
            
            # Add picks
            if len(true_pick_samples) > 0:
                ax3.axvline(true_pick_time, color='red', linestyle='--', linewidth=2, 
                           label='True Pick')
                # Highlight true pick window
                window_start = time_vector[true_pick_samples[0]]
                window_end = time_vector[true_pick_samples[-1]]
                ax3.axvspan(window_start, window_end, alpha=0.2, color='green', 
                           label='True Pick Window')
            
            ax3.axvline(model_pick_time, color='blue', linestyle='-', linewidth=2, 
                       label='Model Pick')
            ax3.axvline(0, color='orange', linestyle=':', alpha=0.7, label='Earthquake Time')
            
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('P-wave Probability')
            ax3.set_title(f'Station {station_info["station"]} - Model Output')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    feature_mode = "Physics-Informed" if use_physics_features else "Raw Waveform"
    plt.suptitle(f'Model Predictions - {feature_mode} Mode', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Model predictions plot saved to {save_path}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════════════

def main(use_physics_features=True, use_shift_gradient=True, num_epochs=25, 
         db_path='seismic_data.db', batch_size=4):
    """
    Main execution function for ML pipeline
    
    Args:
        use_physics_features (bool): Whether to use physics-informed features
        use_shift_gradient (bool): Whether to use shift gradient feature
        num_epochs (int): Number of training epochs
        db_path (str): Path to SQLite database
        batch_size (int): Batch size for training
    """
    
    print("="*80)
    if use_physics_features:
        print("PHYSICS-INFORMED SEISMIC PHASE PICKER - ML PIPELINE")
    else:
        print("RAW WAVEFORM SEISMIC PHASE PICKER - ML PIPELINE")
    print("="*80)
    
    feature_mode = "Physics-Informed" if use_physics_features else "Raw Waveform Only"
    print(f"🔧 Configuration: {feature_mode} mode")
    print(f"🔧 Shift Gradient: {'Enabled' if use_shift_gradient else 'Disabled'}")
    print(f"🕐 Training epochs: {num_epochs}")
    print(f"📊 Database: {db_path}")
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        print("Please run data_mine.py first to create the database.")
        return
    
    # Create datasets
    print('\n')
    print("-" * 50)
    print(f"LOADING DATA FROM DATABASE - {feature_mode.upper()} MODE")
    print("-" * 50)
    
    print("Creating training dataset...")
    train_dataset = SeismicDatabaseDataset(
        db_path=db_path,
        window_size=5,
        use_physics_features=use_physics_features,
        use_shift_gradient=use_shift_gradient,
        train=True
    )
    
    print("Creating validation dataset...")
    val_dataset = SeismicDatabaseDataset(
        db_path=db_path,
        window_size=5,
        use_physics_features=use_physics_features,
        use_shift_gradient=use_shift_gradient,
        train=False
    )
    
    print(f"✅ Training dataset size: {len(train_dataset)}")
    print(f"✅ Validation dataset size: {len(val_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ No data loaded. Please check the database.")
        return
    
    # Visualize features
    print(f"\n📊 VISUALIZING {feature_mode.upper()} FEATURES")
    print("-" * 50)
    
    visualize_features(train_dataset, sample_idx=0, 
                      save_path=f'{"physics_" if use_physics_features else "raw_"}features.png',
                      use_physics_features=use_physics_features)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    print('\n')
    print("-" * 50)
    print(f"INITIALIZING {feature_mode.upper()} MODEL")
    print("-" * 50)
    
    # Get number of input channels from first sample
    sample_features, _ = train_dataset[0]
    n_input_channels = sample_features.shape[0]
    print(f"✅ Input channels (features): {n_input_channels}")
    print(f"✅ Feature shape: {sample_features.shape}")
    
    model = AdaptiveUNet1D(
        in_channels=n_input_channels,
        out_channels=2,
        use_physics_features=use_physics_features
    )
    
    # Initialize feature weights if using physics features
    if use_physics_features:
        model.initialize_feature_weights(n_input_channels)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print('\n')
    print("-" * 50)
    print("🚀 STARTING TRAINING")
    print("-" * 50)
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=num_epochs)
    
    # Load best model
    try:
        model.load_state_dict(torch.load('best_adaptive_model.pth'))
        print("✅ Loaded best model from training")
    except:
        print("⚠️ Could not load best model, using current state")
    
    # Evaluate model
    print("\n📊 EVALUATING MODEL")
    print("-" * 50)
    
    pick_errors = evaluate_picks(model, val_dataset)
    
    # Calculate metrics (assuming 100 Hz sampling rate)
    sampling_rate = 100
    pick_errors_seconds = pick_errors / sampling_rate
    
    print(f"\n🎯 PICK ACCURACY RESULTS - {feature_mode.upper()} MODE")
    print("=" * 60)
    print(f"Mean absolute error: {np.mean(pick_errors_seconds):.4f} ± {np.std(pick_errors_seconds):.4f} seconds")
    print(f"Median absolute error: {np.median(pick_errors_seconds):.4f} seconds")
    print(f"90th percentile error: {np.percentile(pick_errors_seconds, 90):.4f} seconds")
    print(f"95th percentile error: {np.percentile(pick_errors_seconds, 95):.4f} seconds")
    
    # Performance categories
    excellent = np.sum(pick_errors_seconds < 0.5)
    good = np.sum((pick_errors_seconds >= 0.5) & (pick_errors_seconds < 1.0))
    fair = np.sum((pick_errors_seconds >= 1.0) & (pick_errors_seconds < 2.0))
    poor = np.sum(pick_errors_seconds >= 2.0)
    
    print(f"\n📈 Performance Categories:")
    print(f"  Excellent (< 0.5s): {excellent:3d} ({excellent/len(pick_errors_seconds)*100:.1f}%)")
    print(f"  Good (0.5-1.0s):   {good:3d} ({good/len(pick_errors_seconds)*100:.1f}%)")
    print(f"  Fair (1.0-2.0s):   {fair:3d} ({fair/len(pick_errors_seconds)*100:.1f}%)")
    print(f"  Poor (> 2.0s):     {poor:3d} ({poor/len(pick_errors_seconds)*100:.1f}%)")
    
    # Create visualizations
    print("\n📊 CREATING VISUALIZATIONS")
    print("-" * 50)
    
    # Training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {feature_mode}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(pick_errors_seconds, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(pick_errors_seconds), color='red', linestyle='--', 
               label=f'Mean: {np.mean(pick_errors_seconds):.3f}s')
    plt.axvline(np.median(pick_errors_seconds), color='orange', linestyle='--', 
               label=f'Median: {np.median(pick_errors_seconds):.3f}s')
    plt.xlabel('Pick Time Error (seconds)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Pick Time Errors - {feature_mode}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_name = f'{"physics_informed" if use_physics_features else "raw_waveform"}_training_results.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot model predictions
    pred_save_name = f'{"physics_informed" if use_physics_features else "raw_waveform"}_predictions.png'
    plot_model_predictions(model, val_dataset, num_examples=5, 
                          save_path=pred_save_name, use_physics_features=use_physics_features)
    
    # Print final feature weights if using physics features
    if use_physics_features and hasattr(model, 'feature_weights') and model.feature_weights is not None:
        print(f"\n🎛️ Final learned feature weights:")
        feature_names = ['Raw', 'STA/LTA 1', 'Log STA/LTA 1', 'STA/LTA 2', 'Log STA/LTA 2', 
                        'STA/LTA 3', 'Log STA/LTA 3', 'Envelope', 'Env. Deriv.', 'Inst. Freq.',
                        'Low Energy', 'Low Env.', 'Mid Energy', 'Mid Env.', 'High Energy', 'High Env.']
        if use_shift_gradient:
            feature_names.append('Shift Gradient')
        
        weights = model.feature_weights.data.cpu().numpy()
        for i, (name, weight) in enumerate(zip(feature_names[:len(weights)], weights)):
            print(f"  {name:15s}: {weight:.4f}")
    
    # Save model
    model_name = f'{"physics_informed" if use_physics_features else "raw_waveform"}_phase_picker.pth'
    torch.save(model.state_dict(), model_name)
    print(f"\n💾 Model saved as '{model_name}'")
    
    print(f"\n🎉 TRAINING COMPLETE - {feature_mode.upper()} MODE!")
    print("=" * 80)

if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════════════════════════════════
    # 🎯 CONFIGURATION: SET YOUR PREFERENCES HERE
    # ═══════════════════════════════════════════════════════════════════════════════════════
    
    # Toggle physics-informed features ON/OFF
    USE_PHYSICS_FEATURES = True   # Set to False for raw waveform only
    
    # Toggle shift gradient feature ON/OFF
    USE_SHIFT_GRADIENT = True  # Set to False to disable shift gradient feature
    
    # Set number of training epochs
    TRAINING_EPOCHS = 50
    
    # Database path
    DATABASE_PATH = 'seismic_data.db'
    
    # Batch size
    BATCH_SIZE = 4
    
    # Run the main function with your configuration
    main(
        use_physics_features=USE_PHYSICS_FEATURES,
        use_shift_gradient=USE_SHIFT_GRADIENT,
        num_epochs=TRAINING_EPOCHS,
        db_path=DATABASE_PATH,
        batch_size=BATCH_SIZE
    )