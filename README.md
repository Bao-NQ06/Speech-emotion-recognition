# EmoDB Dataset Preprocessing

This repository contains a comprehensive preprocessing pipeline for the EmoDB (Berlin Database of Emotional Speech) dataset, implementing the methods described in recent speech emotion recognition research papers.

## Features

The preprocessing pipeline implements the following techniques:

1. **Pre-emphasis Filter**: Enhances high-frequency components using the FIR filter H(z) = 1 - αz^(-1) where α = 0.97
2. **Signal Normalization**: Z-score normalization using S_Ni = (S_i - μ) / σ
3. **Silent Removal**: Automatically removes silent portions from audio files
4. **Mel-Spectrogram Extraction**: Converts audio to mel-spectrograms with research-based parameters:
   - STFT length: 1024
   - Hop size: 128
   - Window size: 1024 (Hanning window)
   - 128 Mel bins
   - Target sampling rate: 22,050 Hz

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

Run the preprocessing pipeline on the EmoDB dataset:

```bash
python preprocessing.py
```

### Model Training

After preprocessing, train speech emotion recognition models:

**Quick Start (Recommended):**
```bash
python train_simple.py
```

**Advanced Training (Multiple Models):**
```bash
python model.py
```

## Model Architectures

Based on research papers achieving 95%+ accuracy on EmoDB, we implement:

1. **CNN Model**: Convolutional layers for spatial feature extraction from mel-spectrograms
2. **CNN-LSTM Model**: Combines CNN spatial features with LSTM temporal modeling (best performance)
3. **Deep CNN Model**: Deeper architecture for complex feature extraction

### Expected Performance

Research papers show these results on EmoDB:
- CNN models: ~93-94% accuracy
- CNN-LSTM models: ~95-96% accuracy
- Our implementation targets similar performance

## Output Structure

After processing, the following files and directories will be created:

```
processed_data/
├── mel_spectrograms/           # Individual mel-spectrogram images
│   ├── 03a01Fa_mel.png
│   ├── 03a01Nc_mel.png
│   └── ...
├── processed_audio/           # Processed audio files
│   ├── 03a01Fa.wav
│   ├── 03a01Nc.wav
│   └── ...
├── mel_spectrograms.npy       # All mel-spectrograms as numpy array
├── labels.npy                 # Emotion labels as numpy array
└── metadata.csv               # Metadata with filenames and emotions
```

## EmoDB Emotion Labels

The EmoDB dataset uses the following emotion coding:
- **A**: Anger
- **E**: Disgust  
- **F**: Fear
- **T**: Sadness
- **L**: Boredom
- **W**: Happiness
- **N**: Neutral

## Dataset Information

- **Original sampling rates**: Various (automatically converted to 22,050 Hz)
- **Number of files**: ~535 audio files
- **Speakers**: 10 German speakers (5 male, 5 female)
- **File format**: WAV files

## Preprocessing Pipeline Details

1. **Audio Loading**: Files are loaded and resampled to 22,050 Hz
2. **Pre-emphasis**: High-frequency enhancement using FIR filter
3. **Silence Removal**: Removes silent segments using energy-based detection
4. **Normalization**: Z-score normalization of the signal
5. **Feature Extraction**: Mel-spectrogram computation with specified parameters

## Research References

This implementation is based on the preprocessing methods described in:
- "Speech emotion recognition" - Nature Scientific Reports, 2024
- EmoDB: Berlin Database of Emotional Speech

## Requirements

- Python 3.7+
- numpy>=1.21.0
- librosa>=0.9.0
- matplotlib>=3.5.0
- scipy>=1.7.0
- soundfile>=0.10.0
- pandas>=1.3.0
- tqdm>=4.62.0

## Notes

- The script automatically handles the EmoDB filename convention for emotion extraction
- Processed files maintain the original EmoDB naming scheme
- Mel-spectrograms are saved both as individual images and as a combined numpy array for easy loading in machine learning pipelines 