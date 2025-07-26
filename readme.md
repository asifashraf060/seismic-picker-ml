# Physics-Informed P-Wave Picker

A modular, physics-informed machine learning pipeline for seismic P-wave detection using raw waveforms and traditional seismological features. Combines domain expertise with deep learning for enhanced generalization and accuracy.

## Features

- 📡 Seismic data mining from NCEDC (via AWS S3)
- 📚 Physics-informed feature extraction (STA/LTA, frequency energy, envelope, etc.)
- 🔍 Shift gradient for edge emphasis
- 🧠 Adaptive 1D U-Net with attention blocks
- 📈 Training visualization & prediction inspection
- 💾 SQLite database for easy reproducibility

## Quick Start

1. **Mine seismic data** from NCEDC:
```bash
   python data_mine.py
```
2. **Train the model with physics-informed features:
```bash
   python ML-pipeline.py
```
3. Output includes:
- Feature visualizations `(physics_features.png)`
- Model prediction plots `(physics_informed_predictions.png)`
- Training metrics `(physics_informed_training_results.png)`
- Trained model `(physics_informed_phase_picker.pth)`