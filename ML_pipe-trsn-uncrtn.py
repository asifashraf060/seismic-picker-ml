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
import math

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

class MaxAmplitudeFeature:
    @staticmethod
    def compute(waveform: np.ndarray, target_length: int, window_length: np.array, sampling_rate = 100) -> np.ndarray:
        """Compute max amplitude features from a waveform"""

        all_features = []

        for wlen in window_length:
            wlen = int(wlen)

            nyquist_freq = sampling_rate / 2.0
            
            # Define filter frequency band (3-12 Hz is typical for P-wave detection)
            lowcut = 3.0  # Hz
            highcut = 12.0  # Hz
            
            # Normalize frequencies by Nyquist frequency for digital filter
            low_norm = lowcut / nyquist_freq
            high_norm = highcut / nyquist_freq

            # Create filter coefficients
            b, a = signal.butter(4, [low_norm, high_norm], 
                               btype='bandpass', analog=False, output='ba')
            
            # Apply forward-backward filter (zero phase distortion)
            waveform_bt = signal.filtfilt(b, a, waveform)
    
            # 1) Convert all negative values to positive (absolute value)
            abs_waveform = np.abs(waveform_bt)

            # 2) Find the maximum value (peak)
            max_val = np.max(abs_waveform)

            # 3) Divide with its highest values
            nrml = abs_waveform / max_val
            zero_mask = np.where(nrml < 0.01)  # Find indices where normalized value is less than 0.01

            # 4) invert the normalization
            nrml_inv = 1 - nrml
            nrml_inv[zero_mask] = 0  # Set values below 0.01 to zero

            # 5) Calculate the standard deviation in a sliding window
            n = len(nrml_inv)
            mxAmpFt = np.array([
                np.std(nrml_inv[i : i + wlen])
                for i in range(n - wlen + 1)
            ])

            # resize to target_length
            if len(mxAmpFt) > target_length:
                # truncate center
                start = (len(mxAmpFt) - target_length) // 2
                mxAmpFt = mxAmpFt[start:start+target_length]
            elif len(mxAmpFt) < target_length:
                # pad at end
                mxAmpFt = np.pad(mxAmpFt, (0, target_length-len(mxAmpFt)), 'constant')

            all_features.append(mxAmpFt.reshape(1, -1))

        return np.vstack(all_features)


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ TRANSFORMER SELF-ATTENTION COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer self-attention"""
    
    def __init__(self, d_model, max_len=15000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Handle odd d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # For odd dimensions, handle the last dimension separately
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Store as (1, d_model, max_len) for broadcasting
        self.register_buffer('pe', pe.transpose(0, 1).unsqueeze(0))
    
    def forward(self, x):
        # x shape: (batch, channels, length)
        # Add positional encoding directly (broadcasting will handle batch dimension)
        # Only use the required length from positional encoding
        return x + self.pe[:, :x.size(1), :x.size(2)]


class TransformerSelfAttention(nn.Module):
    """
    Memory-efficient Transformer self-attention with sliding window
    Uses local attention to reduce memory consumption
    """
    
    def __init__(self, channels, num_heads=2, dropout=0.1, window_size=256):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size  # Local attention window
        
        # Ensure channels is divisible by num_heads
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        
        self.d_k = channels // num_heads
        
        # Linear transformations for Q, K, V
        self.linear_q = nn.Linear(channels, channels)
        self.linear_k = nn.Linear(channels, channels)
        self.linear_v = nn.Linear(channels, channels)
        
        # Output projection
        self.linear_out = nn.Linear(channels, channels)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(channels)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Feed-forward network (reduced size for memory)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),  # Reduced from 4x to 2x
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def sliding_window_attention(self, Q, K, V):
        """Apply attention with sliding window to reduce memory usage"""
        batch_size, num_heads, seq_len, d_k = Q.shape
        
        # If sequence is short enough, use full attention
        if seq_len <= self.window_size * 2:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            return torch.matmul(attn_weights, V)
        
        # Otherwise, use sliding window attention
        output = torch.zeros_like(V)
        
        # Process in overlapping windows
        stride = self.window_size // 2
        for i in range(0, seq_len, stride):
            end_i = min(i + self.window_size, seq_len)
            
            # Get window queries
            Q_window = Q[:, :, i:end_i]
            
            # Determine key/value window (slightly larger for context)
            start_kv = max(0, i - stride)
            end_kv = min(seq_len, end_i + stride)
            K_window = K[:, :, start_kv:end_kv]
            V_window = V[:, :, start_kv:end_kv]
            
            # Compute attention for this window
            scores = torch.matmul(Q_window, K_window.transpose(-2, -1)) / math.sqrt(d_k)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention and accumulate
            window_output = torch.matmul(attn_weights, V_window)
            
            # Blend overlapping regions
            if i > 0 and i < seq_len - self.window_size:
                # Apply triangular blending for smooth transitions
                blend_size = stride
                blend_weights = torch.linspace(0, 1, blend_size, device=Q.device)
                blend_weights = blend_weights.view(1, 1, -1, 1)
                
                # Blend the overlapping part
                output[:, :, i:i+blend_size] = (
                    output[:, :, i:i+blend_size] * (1 - blend_weights) +
                    window_output[:, :, :blend_size] * blend_weights
                )
                output[:, :, i+blend_size:end_i] = window_output[:, :, blend_size:]
            else:
                output[:, :, i:end_i] = window_output
        
        return output
        
    def forward(self, x):
        # x shape: (batch, channels, length)
        batch_size, channels, seq_len = x.shape
        
        # Downsample if sequence is too long (>2000 samples) - more aggressive
        downsample_factor = 1
        if seq_len > 2000:
            downsample_factor = 4 if seq_len > 4000 else 2
            # Store original for skip connection
            x_original = x
            # Downsample by factor
            x = F.avg_pool1d(x, kernel_size=downsample_factor, stride=downsample_factor)
            seq_len = x.shape[2]
        
        # Add positional encoding
        x_pos = self.pos_encoding(x)
        
        # Transpose for attention: (batch, length, channels)
        x_transposed = x_pos.transpose(1, 2)
        
        # Store residual
        residual = x_transposed
        
        # Compute Q, K, V
        Q = self.linear_q(x_transposed).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.linear_k(x_transposed).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.linear_v(x_transposed).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose for attention computation: (batch, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Apply sliding window attention
        attn_output = self.sliding_window_attention(Q, K, V)
        
        # Concatenate heads: (batch, seq_len, channels)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, channels
        )
        
        # Output projection
        attn_output = self.linear_out(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Add & Norm (residual connection)
        x_attn = self.norm1(residual + attn_output)
        
        # Feed-forward network with residual
        ffn_output = self.ffn(x_attn)
        x_final = self.norm2(x_attn + ffn_output)
        
        # Transpose back: (batch, channels, length)
        x_final = x_final.transpose(1, 2)
        
        # Upsample back to original size if we downsampled
        if downsample_factor > 1:
            x_final = F.interpolate(x_final, size=x_original.shape[2], mode='linear', align_corners=False)
            # Add skip connection with original resolution
            x_final = x_final + x_original
        
        return x_final


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ ENHANCED U-NET WITH TRANSFORMER ATTENTION
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


class AdaptiveUNet1D(nn.Module):
    """
    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │ Memory-Efficient Adaptive 1-D U-Net with Transformer Self-Attention                  │
    │ ──────────────────────────────────────────────────────────────────────────────────── │
    │ • Uses sliding window transformer self-attention to reduce memory usage              │
    │ • Automatic downsampling for very long sequences                                     │
    │ • Multi-head self-attention for capturing long-range dependencies                    │
    | • Uncertainty quantification through dual output heads (pick + std)                  |
    │ • Positional encoding for sequence awareness                                         │
    │ • Feed-forward networks for enhanced feature transformation                          │
    │ • Can work with or without physics-informed features                                 │
    │ • Batch normalization and dropout for better generalization                          │
    └──────────────────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, in_channels=1, out_channels=3, features=[16, 32, 64, 128], 
                 dropout=0.1, use_physics_features=True, num_heads=2, window_size=256):
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
            # Use transformer self-attention with appropriate number of heads
            heads = min(num_heads, feat // 16)  # More conservative: ensure at least 16 dims per head
            heads = max(1, heads)  # At least 1 head
            self.attentions_down.append(TransformerSelfAttention(feat, num_heads=heads, 
                                                                 dropout=dropout, window_size=window_size))
            current_channels = feat
        
        # ============================================
        # 2️⃣ Bottleneck (connects ENCODER & DECODER)
        # ============================================
        self.bottleneck = ConvBlock(features[-1], features[-1]*2, dropout=dropout)
        # Transformer attention for bottleneck
        bottleneck_heads = min(num_heads, (features[-1]*2) // 16)
        bottleneck_heads = max(1, bottleneck_heads)
        self.bottleneck_attention = TransformerSelfAttention(features[-1]*2, num_heads=bottleneck_heads, 
                                                            dropout=dropout, window_size=window_size)
        
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
            # Transformer attention for refined features
            heads = min(num_heads, feat // 16)
            heads = max(1, heads)
            self.attentions_up.append(TransformerSelfAttention(feat, num_heads=heads, 
                                                              dropout=dropout, window_size=window_size))
        
        # ===========================
        # 4️⃣ Final Classification Layer
        # ===========================
        self.final_class = nn.Conv1d(features[0], 2, kernel_size=1)       # per-time class logits (bg vs P)
        self.time_logits = nn.Conv1d(features[0], 1, kernel_size=1)       # per-time density logits (discrete over time)        
        
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
            weighted_x = attention(weighted_x)  # Apply transformer self-attention
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
            
            # Apply transformer self-attention
            attention_idx = idx // 2
            if attention_idx < len(self.attentions_up):
                weighted_x = self.attentions_up[attention_idx](weighted_x)
        
        # classification logits -> per-time softmax over classes
        class_logits = self.final_class(weighted_x)            # (B,2,T)
        classification = F.softmax(class_logits, dim=1)        # (B,2,T)

        # time-density logits -> softmax over time (last dim)
        t_logits = self.time_logits(weighted_x).squeeze(1)     # (B,T)
        q_time = F.softmax(t_logits, dim=-1)                   # (B,T), Σ_t q(t)=1

        # compute soft pick mean and std in sample units
        B, T = q_time.shape
        t_idx = torch.arange(T, device=q_time.device).float()  # (T,)
        mu_hat = (q_time * t_idx).sum(dim=-1, keepdim=True)    # (B,1)
        var_hat = (q_time * (t_idx - mu_hat)**2).sum(dim=-1, keepdim=True) + 1e-6
        sigma_hat = torch.sqrt(var_hat)                        # (B,1)

        # return everything you need
        return classification, q_time, mu_hat, sigma_hat


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
                 use_physics_features=True, use_max_amplitude=True, window_length=None,
                 train=True, train_split=0.8, random_seed=42):
        self.db_path = db_path
        self.window_size = window_size
        self.target_length = target_length
        self.use_physics_features = use_physics_features
        self.use_max_amplitude = use_max_amplitude
        self.window_length = window_length

        # Initialize physics feature extractor if needed
        if self.use_physics_features:
            self.physics_feature_extractor = PhysicsInformedFeatures(sampling_rate=100)
        else:
            self.physics_feature_extractor = None
        
        if self.use_max_amplitude:
            if self.window_length is None:
                raise ValueError("window_length must be provided when use_max_amplitude is True")
            self.max_amplitude_extractor = MaxAmplitudeFeature()
        else:
            self.max_amplitude_extractor = None

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
                physics_features = self.physics_feature_extractor.compute_all_features(waveform_norm)
                final_features = physics_features
            else:
                # Use only raw waveform
                final_features = waveform_norm.reshape(1, -1)
            
            if self.use_max_amplitude and self.max_amplitude_extractor is not None:
                mxAmpFt = self.max_amplitude_extractor.compute(waveform_norm, self.target_length, self.window_length)
                final_features = np.vstack([final_features, mxAmpFt])

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
def _build_time_density_targets(target_batch, T, sigma_tgt_samples=5, device=None):
    """
    Turn binary 1/0 windows into a normalized Gaussian target over time, y(t), per item.
    target_batch: (B, T) LongTensor/FloatTensor with 1s around true pick window
    Returns: y_target (B, T) where sum_t y_target[b,t] = 1
    """
    if device is None:
        device = target_batch.device
    B = target_batch.shape[0]
    t = torch.arange(T, device=device).float()  # (T,)
    y_list = []
    for b in range(B):
        yb = target_batch[b].float()  # (T,)
        idx = torch.where(yb == 1)[0]
        if len(idx) == 0:
            # Fallback if no 1s: put a very broad weak Gaussian at argmax of P-class later
            # (You can refine: here we center at the midpoint to avoid crashing)
            mu_star = (T - 1) / 2.0
        else:
            mu_star = idx.float().mean()
        y = torch.exp(-0.5 * ((t - mu_star) / sigma_tgt_samples) ** 2)
        y = y / (y.sum() + 1e-6)
        y_list.append(y)
    return torch.stack(y_list, dim=0)  # (B, T)

def train_model_with_uncertainty(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the adaptive model with transformer attention and uncertainty quantification"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    fs = 100 # sampling rate (Hz)

    # Loss function with class weighting (background vs P-wave)
    class_weights = torch.FloatTensor([1.0, 3.0]).to(device)  # Higher weight for P-wave
    class_criterion = nn.NLLLoss(weight=class_weights)
    
    # Optimizer with learning rate scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    train_uncertainties = []
    val_uncertainties = []
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_uncertainty = 0.0
        train_batches = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, 
                                                       desc=f'Epoch {epoch+1}/{num_epochs}')):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()

            # Forward pass (after Step 1 changes)
            # classification: (B,2,T)  q_time: (B,T)  mu_hat: (B,1 in samples)  sigma_hat: (B,1 in samples)
            classification, q_time, mu_hat, sigma_hat = model(data)

            # --- (i) Auxiliary segmentation loss (keep your weighted CE) ---
            class_loss = class_criterion((classification.clamp_min(1e-12)).log(), target)

            # --- (ii) Build Gaussian target y(t) from 1/0 window labels ---
            B, _, T = classification.shape
            y_target = _build_time_density_targets(target, T, sigma_tgt_samples=5, device=classification.device)

            # --- (iii) Proper discrete time-density CE:  CE(y || q) = -Σ y(t) log q(t) ---
            # (Clamp to avoid log(0))
            ce_time = -(y_target * (q_time.clamp_min(1e-12).log())).sum(dim=-1).mean()

            # --- (iv) (Step 3 material, if you’re wiring it now) Gaussian NLL on μ̂, σ̂ ---
            # true pick center μ* from the label window
            mu_true = torch.stack([
                (torch.where(target[b] == 1)[0].float().mean()
                if (target[b] == 1).any()
                else torch.argmax(classification[b, 1]).float())
                for b in range(B)
            ], dim=0).unsqueeze(1)  # (B,1) in samples

            sigma_sq = (sigma_hat ** 2).clamp_min(1e-6)
            nll_gauss = 0.5 * (((mu_true - mu_hat) ** 2) / sigma_sq + 2.0 * torch.log(sigma_hat)).mean()

            # --- (v) Total loss (weights you can tune) ---
            total_loss = class_loss + ce_time + nll_gauss
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
            optimizer.step()

            # track mean uncertainty this batch (seconds if you want)
            fs = 100.0  # your pipeline assumes 100 Hz throughout
            train_uncertainty += (sigma_hat.mean().item() / fs)
            train_batches += 1
            train_loss += total_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_uncertainty = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                classification, q_time, mu_hat, sigma_hat = model(data)

                # Compute validation loss
                class_loss = class_criterion((classification.clamp_min(1e-12)).log(), target)

                B, _, T = classification.shape
                y_target = _build_time_density_targets(target, T, sigma_tgt_samples=5, device=classification.device)
                ce_time = -(y_target * (q_time.clamp_min(1e-12).log())).sum(dim=-1).mean()

                mu_true = torch.stack([
                    (torch.where(target[b] == 1)[0].float().mean()
                    if (target[b] == 1).any()
                    else torch.argmax(classification[b, 1]).float())
                    for b in range(B)
                ], dim=0).unsqueeze(1)

                sigma_sq = (sigma_hat ** 2).clamp_min(1e-6)
                nll_gauss = 0.5 * (((mu_true - mu_hat) ** 2) / sigma_sq + 2.0 * torch.log(sigma_hat)).mean()

                val_batch_loss = (class_loss + ce_time + nll_gauss).item()
                val_loss += val_batch_loss
                val_uncertainty += (sigma_hat.mean().item() / fs)
                val_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        avg_train_uncertainty = train_uncertainty / train_batches
        avg_val_uncertainty = val_uncertainty / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_uncertainties.append(avg_train_uncertainty)
        val_uncertainties.append(avg_val_uncertainty)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_transformer-uncertainty_model.pth')
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Train Uncertainty: {avg_train_uncertainty:.4f}, '
              f'Val Uncertainty: {avg_val_uncertainty:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if patience_counter >= 10:
            print("Early stopping triggered")
            break
    
    return train_losses, val_losses, train_uncertainties, val_uncertainties

def choose_sigma_samples(q, mu_smpl, sigma_smpl_scalar, source="moment"):
    """
    Returns σ in samples, either from q(t) (moment) or from the scalar head.
    q: np.ndarray, shape (T,)
    mu_smpl: float, mean in samples
    sigma_smpl_scalar: float, scalar σ from model head (samples)
    """
    if source == "scalar":
        return float(sigma_smpl_scalar)
    # moment: σ = sqrt(E_q[(t-μ)^2])
    t = np.arange(len(q), dtype=np.float32)
    return float(np.sqrt(max(1e-12, (q * (t - mu_smpl) ** 2).sum())))


def confidence_from_q(q, mu_smpl, sigma_samples, mode="mass", k=1.0):
    """
    Confidence from q(t) given μ (samples) and σ (samples).
    mode="point": q(round(μ))
    mode="mass":  ∑_{|t-μ| ≤ kσ} q(t)
    """
    T = len(q)
    if mode == "point":
        idx = int(np.clip(np.rint(mu_smpl), 0, T - 1))
        return float(q[idx])

    # mass in ±kσ, guarantee ≥1 bin
    width = max(1, int(np.ceil(k * max(1e-6, sigma_samples))))
    center = int(np.clip(np.rint(mu_smpl), 0, T - 1))
    lo = max(0, center - width)
    hi = min(T, center + width + 1)
    return float(q[lo:hi].sum())

def evaluate_picks_with_uncertainty(
    model,
    val_dataset,
    fs=100.0,
    confidence_threshold=0.7,
    max_std_threshold=0.5,
    use_sigma_from="moment",      # "moment" or "scalar"
    confidence_mode="mass",       # "mass" or "point"
    mass_k=1.0,                   # k for ±kσ mass
    device=None
):
    """
    Evaluate picks using the new probabilistic outputs.
    - use_sigma_from="moment": σ = sqrt(E_q[(t-μ)^2]) from q(t)
      use_sigma_from="scalar": σ = model's predicted scalar σ̂
    - confidence_mode="mass": confidence = mass within ±kσ around μ
      confidence_mode="point": confidence = q(round(μ))
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    pick_errors = []        # in samples
    uncertainties = []      # in seconds
    rejected_picks = 0
    total_picks = 0

    with torch.no_grad():
        for i in range(len(val_dataset)):
            features, true_label = val_dataset[i]
            # (Optional) per-trace fs from metadata:
            fs_i = float(getattr(val_dataset, "metadata", [{}])[i].get("sampling_rate", fs))
            T = features.shape[1]

            x = features.unsqueeze(0).to(device)
            classification, q_time, mu_hat, sigma_hat = model(x)

            q = q_time[0].detach().cpu().numpy()            # (T,)
            mu_smpl = float(mu_hat[0, 0].detach().cpu())    # samples
            sig_smpl_scalar = float(sigma_hat[0, 0].detach().cpu())  # samples

            # Moment σ from q(t)
            t = np.arange(T, dtype=np.float32)
            sigma_moment = float(np.sqrt(np.clip(((q * (t - mu_smpl) ** 2).sum()), 1e-12, None)))

            # Choose which σ to gate with
            if use_sigma_from.lower() == "scalar":
                sigma_samples = sig_smpl_scalar
            else:  # "moment"
                sigma_samples = sigma_moment

            sigma_seconds = sigma_samples / fs_i

            # Confidence
            sigma_samples = choose_sigma_samples(q, mu_smpl, sig_smpl_scalar, source=use_sigma_from.lower())
            sigma_seconds = sigma_samples / fs_i
            conf = confidence_from_q(q, mu_smpl, sigma_samples, mode=confidence_mode.lower(), k=mass_k)

            total_picks += 1

            # Gate by confidence + uncertainty
            if (conf < confidence_threshold) or (sigma_seconds > max_std_threshold):
                rejected_picks += 1
                continue

            # Truth center from 1-window label
            tl = true_label.numpy()
            idx = np.where(tl == 1)[0]
            if len(idx) > 0:
                true_mu = float(idx.mean())
                err_samples = abs(mu_smpl - true_mu)
                pick_errors.append(err_samples)
                uncertainties.append(sigma_seconds)

    acceptance_rate = (total_picks - rejected_picks) / total_picks if total_picks > 0 else 0.0
    return np.array(pick_errors), np.array(uncertainties), acceptance_rate, rejected_picks

def visualize_features(dataset, save_path='features_transformer', use_physics_features=True,
                       use_max_amplitude=True, window_length=None):
    """Visualize the features for all the samples"""
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for sample_idx in range(len(dataset)):

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

        # Account for max amplitude feature if present
        if use_max_amplitude:
            for i in range(len(window_length)):
                feature_names.append(f'Max Filter WL:{int(window_length[i])}')

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
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/features_of_seismogram{sample_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Features visualization saved to {save_path}")


def plot_predictions_uncertainty(model, dataset, num_examples=1, 
                                 save_path='predictions_transformer_uncertainty',
                                 use_physics_features=True,
                                 fs=100.0,
                                 confidence_threshold=0.7,
                                 max_std_threshold=0.5,
                                 use_sigma_from="moment",
                                 confidence_mode="mass",
                                 mass_k=1.0):
    """Plot model predictions with the SAME gating as the evaluator."""
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(num_examples):
            plt.figure(figsize=(20, 5))

            features, true_label = dataset[i]
            station_info = dataset.metadata[i]

            features_batch = features.unsqueeze(0).to(device)
            classification, q_time, mu_hat, sigma_hat = model(features_batch)

            q = q_time[0].cpu().numpy()
            mu_smpl = float(mu_hat[0, 0].cpu())
            sig_smpl_scalar = float(sigma_hat[0, 0].cpu())

            # Time vector
            sampling_rate = station_info.get('sampling_rate', fs)
            fs_i = float(sampling_rate)
            pre_time, post_time = 3, 120
            T = features.shape[1]
            time_vector = np.linspace(-pre_time, post_time, T)

            # Raw waveform for panel 1
            raw_waveform = features[0].numpy()

            # ---------------- Panel 1: Raw waveform + gating result ----------------
            plt.subplot(1, 4, 1)
            plt.plot(time_vector, raw_waveform, 'k-', linewidth=0.8, alpha=0.8)

            # Shared logic for σ + confidence
            sigma_samples = choose_sigma_samples(q, mu_smpl, sig_smpl_scalar, source=use_sigma_from.lower())
            sigma_seconds = sigma_samples / fs_i
            conf = confidence_from_q(q, mu_smpl, sigma_samples, mode=confidence_mode.lower(), k=mass_k)

            pred_idx = int(np.clip(np.rint(mu_smpl), 0, T - 1))
            model_pick_time = time_vector[pred_idx]

            # Truth line
            true_pick_samples = np.where(true_label.numpy() == 1)[0]
            if len(true_pick_samples) > 0:
                true_pick_time = time_vector[int(np.mean(true_pick_samples))]
                plt.axvline(true_pick_time, color='red', linestyle='--', linewidth=2.5,
                            label=f'True Pick ({true_pick_time:.2f}s)')

            # Gating (identical policy as evaluator)
            accepted = (conf >= confidence_threshold) and (sigma_seconds <= max_std_threshold)
            pick_color = 'green' if accepted else 'orange'
            pick_label = f'Model Pick ({model_pick_time:.2f}s) – {"ACCEPTED" if accepted else "REJECTED"}'
            plt.axvline(model_pick_time, color=pick_color, linestyle=':', linewidth=2, label=pick_label)
            plt.axvline(0, color='gray', linestyle=':', alpha=0.7, label='Earthquake Time')
            plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
            plt.title(f'Station {station_info.get("station","?")} – Raw Waveform')
            plt.legend(fontsize=8); plt.grid(True, alpha=0.3)

            # ---------------- Panel 2: q(t) with μ̂ ± 1σ ----------------
            plt.subplot(1, 4, 2)
            mu_time = time_vector[pred_idx]
            plt.plot(time_vector, q, linewidth=2, label='q(t) (time-density)')
            plt.axvline(mu_time, color='k', linestyle=':', linewidth=2, label='μ̂')
            plt.axvspan(mu_time - sigma_seconds, mu_time + sigma_seconds, alpha=0.2, label='±1σ')
            plt.ylim(0, 1); plt.xlabel('Time (s)'); plt.ylabel('Density')
            plt.legend(fontsize=8); plt.grid(True, alpha=0.3)

            # ---------------- Panel 3: σ (seconds) ----------------
            plt.subplot(1, 4, 3)
            plt.plot(time_vector, np.zeros_like(time_vector) + sigma_seconds, linewidth=2, label='σ (seconds)')
            plt.axhline(y=max_std_threshold, color='orange', linestyle='--', label=f'Max σ ({max_std_threshold}s)')
            plt.xlabel('Time (s)'); plt.ylabel('Uncertainty (s)')
            plt.legend(fontsize=8); plt.grid(True, alpha=0.3)

            # ---------------- Panel 4: combined view ----------------
            plt.subplot(1, 4, 4)
            wf_norm = raw_waveform / (np.max(np.abs(raw_waveform)) + 1e-10)
            plt.plot(time_vector, wf_norm, linewidth=0.5, alpha=0.5, label='Waveform')
            plt.plot(time_vector, q, linewidth=2, label='q(t)')
            plt.axvline(mu_time, color='k', linestyle=':', linewidth=2, label='μ̂')
            plt.axvspan(mu_time - sigma_seconds, mu_time + sigma_seconds, alpha=0.1,
                        label=f'±1σ ({sigma_seconds:.3f}s)')
            plt.ylim(-1.2, 1.2); plt.legend(fontsize=8, loc='upper right'); plt.grid(True, alpha=0.3)

            plt.suptitle(f"Model Predictions with Uncertainty Quantification - Station {station_info.get('station','?')}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{save_path}/predictions_uncertainty_{i}.png', dpi=150, bbox_inches='tight')
            plt.close()


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════════════

def main(use_physics_features=True, use_max_amplitude=True, window_length=None,
         num_epochs=25, num_heads=2, window_size=256,
         db_path='seismic_data.db', batch_size=1, 
         confidence_threshold=0.7, max_std_threshold=0.5):
    """
    Main execution function for ML pipeline with Transformer Self-Attention and Uncertainty Quantification
    
    Args:
        use_physics_features (bool): Whether to use physics-informed features
        use_max_amplitude (bool): Whether to use max amplitude feature
        window_length (array): Window lengths for max amplitude feature
        num_epochs (int): Number of training epochs
        num_heads (int): Number of attention heads in transformer
        window_size (int): Size of attention window for sliding window attention
        db_path (str): Path to SQLite database
        batch_size (int): Batch size for training
        confidence_threshold (float): Minimum P-wave probability to accept a pick
        max_std_threshold (float): Maximum uncertainty (seconds) to accept a pick
    """
    
    print("="*80)
    print("TRANSFORMER SELF-ATTENTION SEISMIC PHASE PICKER WITH UNCERTAINTY QUANTIFICATION")
    print("="*80)
    
    feature_mode = "Physics-Informed" if use_physics_features else "Raw Waveform Only"
    print(f"🔧 Configuration: {feature_mode} mode")
    print(f"🔧 Max Amplitude: {'Enabled' if use_max_amplitude else 'Disabled'}")
    print(f"🔧 Transformer Heads: {num_heads}")
    print(f"🔧 Uncertainty Quantification: ENABLED")
    print(f"🔧 Confidence Threshold: {confidence_threshold}")
    print(f"🔧 Max Uncertainty Threshold: {max_std_threshold} seconds")
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
        use_max_amplitude=use_max_amplitude,
        window_length=window_length,
        train=True
    )

    print("Creating validation dataset...")
    val_dataset = SeismicDatabaseDataset(
        db_path=db_path,
        window_size=5,
        use_physics_features=use_physics_features,
        use_max_amplitude=use_max_amplitude,
        window_length=window_length,
        train=False
    )

    print(f"✅ Training dataset size: {len(train_dataset)}")
    print(f"✅ Validation dataset size: {len(val_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ No data loaded. Please check the database.")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    print('\n')
    print("-" * 50)
    print(f"INITIALIZING TRANSFORMER {feature_mode.upper()} MODEL WITH UNCERTAINTY")
    print("-" * 50)
    
    # Get number of input channels from first sample
    sample_features, _ = train_dataset[0]
    n_input_channels = sample_features.shape[0]
    print(f"✅ Input channels (features): {n_input_channels}")
    print(f"✅ Feature shape: {sample_features.shape}")
    print(f"✅ Transformer attention heads: {num_heads}")
    print(f"✅ Heads: class(bg/P) + time-density q(t) → μ̂, σ̂")
    print(f"🔧 Eval gating → confidence≥{confidence_threshold}, σ≤{max_std_threshold}s, "
      f"σ_from={'moment'}; confidence_mode={'mass'}")
    
    model = AdaptiveUNet1D(
        in_channels=n_input_channels,
        out_channels=3,  # Changed from 2 to 3 for uncertainty
        use_physics_features=use_physics_features,
        num_heads=num_heads,
        window_size=window_size
    )
    
    # Initialize feature weights if using physics features
    if use_physics_features:
        model.initialize_feature_weights(n_input_channels)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model with uncertainty
    print('\n')
    print("-" * 50)
    print("🚀 STARTING TRAINING WITH TRANSFORMER SELF-ATTENTION AND UNCERTAINTY")
    print("-" * 50)
    
    train_losses, val_losses, train_uncertainties, val_uncertainties = train_model_with_uncertainty(
        model, train_loader, val_loader, num_epochs=num_epochs)
    
    # Load best model
    try:
        model.load_state_dict(torch.load('best_transformer-uncertainty_model.pth'))
        print("✅ Loaded best transformer model with uncertainty from training")
    except:
        print("⚠️ Could not load best model, using current state")
    
    # Evaluate model with uncertainty filtering
    print("\n📊 EVALUATING TRANSFORMER MODEL WITH UNCERTAINTY FILTERING")
    print("-" * 50)
    
    pick_errors, uncertainties, acceptance_rate, rejected_picks = evaluate_picks_with_uncertainty(
        model, val_dataset,
        fs=100.0,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_std_threshold=MAX_STD_THRESHOLD,
        use_sigma_from="moment",         # or "scalar"
        confidence_mode="mass",
        mass_k=1.0
    )
    
    # Calculate metrics (assuming 100 Hz sampling rate)
    sampling_rate = 100
    
    if len(pick_errors) > 0:
        pick_errors_seconds = pick_errors / sampling_rate
        
        # --- Calibration metric (ΔZ) ---
        # err_seconds: array of errors in seconds
        # uncertainties: array of predicted σ in seconds (already from evaluate)
        err_seconds = pick_errors_seconds
        sigma_seconds = uncertainties
        delta_z = np.mean((err_seconds / (sigma_seconds + 1e-9))**2) - 1.0
        print(f"Calibration ΔZ (target ≈ 0): {delta_z:.3f}")

        print(f"\n🎯 PICK ACCURACY RESULTS - TRANSFORMER WITH UNCERTAINTY FILTERING")
        print("=" * 60)
        print(f"Acceptance Rate: {acceptance_rate:.1%} ({len(pick_errors)}/{len(val_dataset)} picks accepted)")
        print(f"Rejected Picks: {rejected_picks} (low confidence or high uncertainty)")
        print(f"\nFor Accepted Picks:")
        print(f"Mean absolute error: {np.mean(pick_errors_seconds):.4f} ± {np.std(pick_errors_seconds):.4f} seconds")
        print(f"Median absolute error: {np.median(pick_errors_seconds):.4f} seconds")
        print(f"90th percentile error: {np.percentile(pick_errors_seconds, 90):.4f} seconds")
        print(f"95th percentile error: {np.percentile(pick_errors_seconds, 95):.4f} seconds")
        print(f"Mean uncertainty: {np.mean(uncertainties):.4f} seconds")
        
        # Performance categories
        excellent = np.sum(pick_errors_seconds < 0.5)
        good = np.sum((pick_errors_seconds >= 0.5) & (pick_errors_seconds < 1.0))
        fair = np.sum((pick_errors_seconds >= 1.0) & (pick_errors_seconds < 2.0))
        poor = np.sum(pick_errors_seconds >= 2.0)
        
        print(f"\n📈 Performance Categories (Accepted Picks):")
        print(f"  Excellent (< 0.5s): {excellent:3d} ({excellent/len(pick_errors_seconds)*100:.1f}%)")
        print(f"  Good (0.5-1.0s):   {good:3d} ({good/len(pick_errors_seconds)*100:.1f}%)")
        print(f"  Fair (1.0-2.0s):   {fair:3d} ({fair/len(pick_errors_seconds)*100:.1f}%)")
        print(f"  Poor (> 2.0s):     {poor:3d} ({poor/len(pick_errors_seconds)*100:.1f}%)")
    else:
        print("⚠️ No picks were accepted with current confidence/uncertainty thresholds")
        print(f"Consider adjusting thresholds (current: confidence>{confidence_threshold}, std<{max_std_threshold}s)")
    
    # Create visualizations
    print("\n📊 CREATING VISUALIZATIONS WITH UNCERTAINTY")
    print("-" * 50)
    
    if len(pick_errors) > 0:
        # Training curves with uncertainty
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
        axes[0, 0].plot(val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Uncertainty curves
        axes[0, 1].plot(train_uncertainties, label='Training Uncertainty', color='green')
        axes[0, 1].plot(val_uncertainties, label='Validation Uncertainty', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Mean Uncertainty')
        axes[0, 1].set_title('Model Uncertainty During Training')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error distribution
        axes[1, 0].hist(pick_errors_seconds, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(np.mean(pick_errors_seconds), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(pick_errors_seconds):.3f}s')
        axes[1, 0].axvline(np.median(pick_errors_seconds), color='orange', linestyle='--', 
                   label=f'Median: {np.median(pick_errors_seconds):.3f}s')
        axes[1, 0].set_xlabel('Pick Time Error (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Error Distribution ({acceptance_rate:.1%} Accepted)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Uncertainty vs Error scatter plot
        axes[1, 1].scatter(uncertainties, pick_errors_seconds, alpha=0.5, color='purple')
        axes[1, 1].set_xlabel('Predicted Uncertainty (seconds)')
        axes[1, 1].set_ylabel('Actual Error (seconds)')
        axes[1, 1].set_title('Uncertainty Calibration')
        # Add diagonal line for perfect calibration
        max_val = max(np.max(uncertainties), np.max(pick_errors_seconds))
        axes[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Calibration')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_name = f'transformer_uncertainty_{"physics_informed" if use_physics_features else "raw_waveform"}_results.png'
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.show()
    
    # Plot model predictions with uncertainty
    plot_predictions_uncertainty(
        model, val_dataset, 
        num_examples=min(10, len(val_dataset)), 
        save_path='predictions_transformer_uncertainty',
        use_physics_features=use_physics_features,
        confidence_threshold=confidence_threshold,
        max_std_threshold=max_std_threshold
    )

    # Print final feature weights if using physics features
    if use_physics_features and hasattr(model, 'feature_weights') and model.feature_weights is not None:
        print(f"\n🎛️ Final learned feature weights (Transformer with Uncertainty):")
        feature_names = ['Raw', 'STA/LTA 1', 'Log STA/LTA 1', 'STA/LTA 2', 'Log STA/LTA 2', 
                        'STA/LTA 3', 'Log STA/LTA 3', 'Envelope', 'Env. Deriv.', 'Inst. Freq.',
                        'Low Energy', 'Low Env.', 'Mid Energy', 'Mid Env.', 'High Energy', 'High Env.']
        if use_max_amplitude:
            for wl in window_length:
                feature_names.append(f'Max Amp WL:{int(wl)}')

        weights = model.feature_weights.data.cpu().numpy()
        for i, (name, weight) in enumerate(zip(feature_names[:len(weights)], weights)):
            print(f"  {name:15s}: {weight:.4f}")
    
    # Save model
    model_name = f'transformer_uncertainty_{"physics_informed" if use_physics_features else "raw_waveform"}_phase_picker.pth'
    torch.save(model.state_dict(), model_name)
    print(f"\n💾 Model saved as '{model_name}'")
    
    # Print summary of uncertainty quantification
    print(f"\n📊 UNCERTAINTY QUANTIFICATION SUMMARY")
    print("=" * 60)
    print(f"The model now provides uncertainty estimates for each prediction.")
    print(f"Picks are automatically filtered based on:")
    print(f"  • Confidence threshold: {confidence_threshold} (P-wave probability)")
    print(f"  • Maximum uncertainty: {max_std_threshold} seconds")
    print(f"\nThis filtering improved pick quality by rejecting {rejected_picks}/{len(val_dataset)} uncertain picks.")
    print(f"Adjust thresholds to balance between accuracy and completeness.")
    
    print(f"\n🎉 TRAINING COMPLETE - TRANSFORMER WITH UNCERTAINTY QUANTIFICATION!")
    print("=" * 80)


if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════════════════════════════════
    # 🎯 CONFIGURATION: SET YOUR PREFERENCES HERE
    # ═══════════════════════════════════════════════════════════════════════════════════════
    
    # Toggle physics-informed features ON/OFF
    USE_PHYSICS_FEATURES = True   # Set to False for raw waveform only

    # Toggle max amplitude feature ON/OFF
    USE_MAX_AMPLITUDE = True  # Set to False to disable max amplitude feature
    window_length = np.array([100, 200, 500])  # Window lengths for max amplitude feature

    # Transformer configuration
    NUM_HEADS = 2  # Minimal attention heads for memory efficiency
    WINDOW_SIZE = 256  # Smaller sliding window for attention
    
    # Uncertainty quantification parameters
    CONFIDENCE_THRESHOLD = 0.65  # Minimum P-wave probability to accept a pick
    MAX_STD_THRESHOLD = 2  # Maximum uncertainty in seconds to accept a pick

    # Set number of training epochs
    TRAINING_EPOCHS = 5
    
    # Database path
    DATABASE_PATH = 'seismic_data_3.db'
    
    # Batch size - minimal for memory efficiency
    BATCH_SIZE = 1
    
    # Run the main function with your configuration
    main(
        use_physics_features=USE_PHYSICS_FEATURES,
        use_max_amplitude=USE_MAX_AMPLITUDE,
        window_length=window_length,
        num_epochs=TRAINING_EPOCHS,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        db_path=DATABASE_PATH,
        batch_size=BATCH_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_std_threshold=MAX_STD_THRESHOLD
    )