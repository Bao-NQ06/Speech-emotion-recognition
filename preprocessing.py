#!/usr/bin/env python3
"""
EmoDB Preprocessing with MFCC Features
Uses MFCC features with delta and delta-delta features as per user specifications:
- MFCC with 128 bands and 128 frames
- Overlapping windowing approach
- Delta and delta-delta features (3-channel output)
- Target sample rate: 44100 Hz
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class EmoDBMFCCPreprocessor:
    def __init__(self, data_path="emoDB/wav", output_path="corrected_data", target_sr=44100, 
                 bands=128, frames=128, hop_length=512, n_fft=1024):
        """
        Initialize preprocessor with MFCC-based feature extraction
        
        Parameters:
        - bands: Number of MFCC coefficients (128)
        - frames: Number of frames per window (128)  
        - hop_length: Distance between frames (512)
        - n_fft: FFT window size (1024)
        - target_sr: Target sample rate (44100)
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.target_sr = target_sr
        self.bands = bands
        self.frames = frames
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # Calculate window size for overlapping windows
        self.window_size = hop_length * (frames - 1)
        
        # Emotion mapping for EmoDB
        self.emotion_map = {
            'A': 'angry',    # anger -> angry
            'E': 'disgust',  # disgust -> disgust  
            'F': 'fear',     # fear -> fear
            'T': 'sad',      # sadness -> sad
            'L': 'boredom',  # boredom -> boredom
            'W': 'happy',    # happiness -> happy
            'N': 'neutral'   # neutral -> neutral
        }
        
        # Label to index mapping
        self.label_to_index = {
            'fear': 0,
            'disgust': 1,
            'happy': 2,
            'angry': 3,
            'sad': 4,
            'neutral': 5,
            'boredom': 6
        }
        
        # Create output directories
        self.output_path.mkdir(exist_ok=True)
        (self.output_path / "mel_spectrograms").mkdir(exist_ok=True)
        (self.output_path / "processed_audio").mkdir(exist_ok=True)
        
        print("=== EmoDB MFCC Preprocessor ===")
        print("Feature Extraction Specifications:")
        print(f"  MFCC bands: {self.bands}")
        print(f"  Frames per window: {self.frames}")
        print(f"  Hop length: {self.hop_length}")
        print(f"  FFT size: {self.n_fft}")
        print(f"  Window size: {self.window_size} samples")
        print(f"  Target sample rate: {self.target_sr} Hz")
        print(f"  Features: MFCC + Delta + Delta-Delta (3 channels)")

    def get_sound_data(self, path, target_sr=None):
        """
        Input:
          - path: path to audio file
          - target_sr: target sample rate
        Output:
          - 1st element: sound data (array of waveform values), 1D if mono,
                         2D (frames x channels) if stereo
          - 2nd element: sample rate (the current sample rate)
        """
        if target_sr is None:
            target_sr = self.target_sr
            
        data, orig_sr = sf.read(path)
        data_resample = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)

        # Get average if audio is multi channels
        if len(data_resample.shape) > 1:
            data_resample = np.average(data_resample, axis=1)
        return data_resample, target_sr

    def windows(self, data, window_size):
        """
        Break the sample into smaller, half overlapping and equal samples
        to satisfy the balance in each sample's length.
        Input:
          - data: sound data
          - window_size: size of each smaller sample
        Yield:
          - start: start index of each sample
          - end: end index of each sample
        """
        start = 0
        while start + window_size < len(data):
            yield start, start + window_size
            start += (window_size // 2)

    def get_emotion_label(self, filename):
        """
        Extract emotion label from EmoDB filename convention
        """
        # Extract emotion from filename (5th character in EmoDB naming)
        emotion_code = filename[5]
        return self.emotion_map.get(emotion_code, 'unknown')

    def extract_features_from_file(self, file_path):
        """
        Extract MFCC features with delta and delta-delta from a single file
        """
        try:
            # Get emotion label
            emotion = self.get_emotion_label(file_path.name)
            if emotion == 'unknown':
                print(f"Unknown emotion for {file_path.name}")
                return None
                
            label_index = self.label_to_index[emotion]
            
            # Load audio data
            data, sr = self.get_sound_data(file_path, target_sr=self.target_sr)
            
            print(f"Processing: {file_path.name} - {emotion}")
            print(f"  Audio length: {len(data)} samples ({len(data)/sr:.2f}s)")
            
            # Extract features from overlapping windows
            file_features = []
            file_labels = []
            
            window_count = 0
            for start, end in self.windows(data, self.window_size):
                signal = data[start:end]
                
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(
                    y=signal, 
                    sr=sr, 
                    n_mfcc=self.bands,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                
                # Ensure we have exactly the right number of frames
                if mfcc.shape[1] != self.frames:
                    if mfcc.shape[1] > self.frames:
                        # Truncate
                        mfcc = mfcc[:, :self.frames]
                    else:
                        # Pad
                        pad_width = self.frames - mfcc.shape[1]
                        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                
                file_features.append(mfcc)
                file_labels.append(label_index)
                window_count += 1
            
            print(f"  Extracted {window_count} windows")
            
            return {
                'features': file_features,
                'labels': file_labels,
                'emotion': emotion,
                'filename': file_path.name,
                'window_count': window_count
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def process_dataset(self, save_images=True):
        """
        Process entire EmoDB dataset with MFCC feature extraction
        """
        print("Starting EmoDB dataset preprocessing with MFCC features...")
        print("Using overlapping windowing and delta features\n")
        
        # Get all wav files
        wav_files = list(self.data_path.glob("*.wav"))
        
        if not wav_files:
            print(f"No WAV files found in {self.data_path}")
            return None
        
        print(f"Found {len(wav_files)} audio files")
        
        # Initialize data storage
        all_features = []
        all_labels = []
        all_filenames = []
        processed_files = []
        
        # Process each file
        for file_path in tqdm(wav_files, desc="Extracting MFCC features"):
            result = self.extract_features_from_file(file_path)
            
            if result is not None:
                all_features.extend(result['features'])
                all_labels.extend(result['labels'])
                all_filenames.extend([result['filename']] * result['window_count'])
                processed_files.append({
                    'filename': result['filename'],
                    'emotion': result['emotion'],
                    'windows': result['window_count']
                })
        
        print(f"\nFeature extraction completed!")
        print(f"Total windows extracted: {len(all_features)}")
        
        # Convert to numpy arrays and reshape
        mfcc_features = np.asarray(all_features)
        print(f"MFCC features shape: {mfcc_features.shape}")
        
        # Reshape to 4D: (n_windows, n_bands, n_frames, n_features)
        mfcc_features = mfcc_features.reshape(len(all_features), self.bands, self.frames, 1)
        
        # Create 3-channel feature map (MFCC, Delta, Delta-Delta)
        features = np.concatenate((
            mfcc_features, 
            np.zeros(np.shape(mfcc_features)), 
            np.zeros(np.shape(mfcc_features))
        ), axis=3)
        
        # Calculate delta and delta-delta features
        print("Computing delta and delta-delta features...")
        for i in tqdm(range(features.shape[0]), desc="Computing deltas"):
            # Delta (1st derivative)
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
            # Delta-Delta (2nd derivative)  
            features[i, :, :, 2] = librosa.feature.delta(features[i, :, :, 0], order=2)
        
        # Convert labels to numpy array
        labels = np.array(all_labels)
        
        # Save processed data
        np.save(self.output_path / "mel_spectrograms.npy", features)  # Keep same name for compatibility
        np.save(self.output_path / "labels.npy", labels)
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame({
            'filename': all_filenames,
            'emotion': [self.get_emotion_from_index(label) for label in all_labels]
        })
        metadata_df.to_csv(self.output_path / "metadata.csv", index=False)
        
        # Save file summary
        file_summary_df = pd.DataFrame(processed_files)
        file_summary_df.to_csv(self.output_path / "file_summary.csv", index=False)
        
        # Save sample visualizations
        if save_images and len(features) > 0:
            self.save_feature_visualizations(features[:10], all_labels[:10], all_filenames[:10])
        
        print(f"\n✅ MFCC preprocessing completed!")
        print(f"Final feature shape: {features.shape}")
        print(f"Features: {self.bands} MFCC bands × {self.frames} frames × 3 channels")
        print(f"Total windows: {len(features)}")
        print(f"Output saved to: {self.output_path}")
        
        # Print emotion distribution
        emotion_counts = metadata_df['emotion'].value_counts()
        print("\nEmotion distribution (windows):")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} windows")
        
        # Print file distribution
        file_emotion_counts = file_summary_df['emotion'].value_counts()
        print("\nEmotion distribution (files):")
        for emotion, count in file_emotion_counts.items():
            print(f"  {emotion}: {count} files")
        
        return features, labels

    def get_emotion_from_index(self, index):
        """Convert label index back to emotion name"""
        index_to_label = {v: k for k, v in self.label_to_index.items()}
        return index_to_label.get(index, 'unknown')

    def save_feature_visualizations(self, features, labels, filenames):
        """Save sample MFCC feature visualizations"""
        for i, (feature, label, filename) in enumerate(zip(features[:5], labels[:5], filenames[:5])):
            emotion = self.get_emotion_from_index(label)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # MFCC
            axes[0].imshow(feature[:, :, 0], aspect='auto', origin='lower', cmap='viridis')
            axes[0].set_title(f'MFCC - {emotion} - {filename}')
            axes[0].set_xlabel('Time Frames')
            axes[0].set_ylabel('MFCC Coefficients')
            
            # Delta
            axes[1].imshow(feature[:, :, 1], aspect='auto', origin='lower', cmap='viridis')
            axes[1].set_title(f'Delta - {emotion} - {filename}')
            axes[1].set_xlabel('Time Frames')
            axes[1].set_ylabel('MFCC Coefficients')
            
            # Delta-Delta
            axes[2].imshow(feature[:, :, 2], aspect='auto', origin='lower', cmap='viridis')
            axes[2].set_title(f'Delta-Delta - {emotion} - {filename}')
            axes[2].set_xlabel('Time Frames')
            axes[2].set_ylabel('MFCC Coefficients')
            
            plt.tight_layout()
            
            # Save image
            image_path = self.output_path / "mel_spectrograms" / f"{filename[:-4]}_mfcc_features.png"
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()

def extract_features(df, label_dict, bands=128, frames=128, hop_length=512, n_fft=1024, target_sr=44100):
    """
    Extract features using the user's original function signature
    Input:
      - df: DataFrame with 'Path' and 'Emotions' columns
      - label_dict: Dictionary mapping emotions to indices
      - frames: Number of frames in each window
      - n_fft: Length of each frame
      - hop_length: Distance from the start index of current frame to the next one
      - bands: Number of mel bands to generate
    Output:
      - features: 4D array (n_windows, n_bands, n_frames, n_features)
      - labels: 1D array of class labels
    """
    # Declare features's variables
    window_size = hop_length * (frames - 1)
    mfcc_specgrams = []
    class_labels = []

    def get_sound_data(path, target_sr=44100):
        """
        Input:
          - path: path to audio file
          - sr: target sample rate
        Output:
          - 1st element: sound data (array of waveform values), 1D if mono,
                         2D (frames x channels) if stero
          - 2nd element: sample rate (the current sample rate)
        """
        data, orig_sr = sf.read(path)
        data_resample = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)

        # Get average if audio is muilti channels
        if len(data_resample.shape) > 1:
            data_resample = np.average(data_resample, axis=1)
        return data_resample, target_sr

    def windows(data, window_size):
        """
        Breake the sample into smaller, half overlaping and equal samples
        to satisfy the balanced in each sample's length.
        Input:
          - data: sound data
          - window_size: size of each smaller sample
        Yield:
          - start: start index of each sample
          - end: end index of each sample
        """
        start = 0
        while start + window_size < len(data):
            yield start, start + window_size
            start += (window_size // 2)

    for row in tqdm(df.itertuples()):
        if row.Path.endswith(".wav"):
            label = label_dict[row.Emotions]
            
            # Get each feature
            data, sr = get_sound_data(row.Path, target_sr=target_sr)
            for start, end in windows(data, window_size):
                signal = data[start:end]

                # Add label
                class_labels.append(label)

                # Get mfcc feature
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=bands, 
                                          n_fft=n_fft, hop_length=hop_length)
                
                # Ensure correct frame count
                if mfcc.shape[1] != frames:
                    if mfcc.shape[1] > frames:
                        mfcc = mfcc[:, :frames]
                    else:
                        pad_width = frames - mfcc.shape[1]
                        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                
                mfcc_specgrams.append(mfcc)
                # mfcc's shape is (n_mfcc, n_frames)

    # Reshape 4D input (n_windows, n_bands, n_frames, n_features)
    mfcc_specgrams = np.asarray(mfcc_specgrams).reshape(len(mfcc_specgrams), bands, frames, 1)

    # Combined USED features into a feature map
    features = np.concatenate((mfcc_specgrams, np.zeros(np.shape(mfcc_specgrams)), np.zeros(np.shape(mfcc_specgrams))), axis=3)

    # Create the third feature map which is the delta (derivative) of the log-scaled mel-spectrogram - USED
    for i in range(features.shape[0]):
        features[i, :, :, 2] = librosa.feature.delta(features[i, :, :, 0])
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0], order=2)

    return np.array(features), np.array(class_labels)

# Label to index mapping as provided by user
label_to_index = {
    'fear': 0,
    'disgust': 1,
    'happy': 2,
    'angry': 3,
    'sad': 4,
    'neutral': 5,
    'boredom': 6
}

def main():
    """
    Main function to run MFCC-based preprocessing
    """
    # Initialize MFCC preprocessor with user specifications
    preprocessor = EmoDBMFCCPreprocessor(
        data_path="emoDB/wav",
        output_path="corrected_data", 
        target_sr=44100,
        bands=128,
        frames=128,
        hop_length=512,
        n_fft=1024
    )
    
    # Process dataset with MFCC features
    features, labels = preprocessor.process_dataset(save_images=True)
    
    if features is not None:
        print(f"\n🎉 Success! Generated MFCC features with shape: {features.shape}")
        print(f"Features: {features.shape[1]} bands × {features.shape[2]} frames × {features.shape[3]} channels")
        print(f"Use 'corrected_data/' for model training.")
    else:
        print("❌ Preprocessing failed!")

if __name__ == "__main__":
    main() 