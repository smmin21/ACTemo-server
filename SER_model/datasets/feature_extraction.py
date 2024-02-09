import warnings
import librosa
import pandas as pd
import numpy as np
import tqdm as tqdm
import torchaudio
import os
import pdb
from time import time



def extract_features(data, sample_rate):
    # ZCR
    # zcr_time = time()
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally
    # print(f"ZCR Time: {time() - zcr_time}")

    # Chroma_stft
    # chroma_time = time()
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally
    # print(f"Chroma Time: {time() - chroma_time}")

    # MFCC
    # mfcc_time = time()
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally
    # print(f"MFCC Time: {time() - mfcc_time}")

    # MelSpectogram
    # mel_time = time()
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    # print(f"Mel Time: {time() - mel_time}")
    
    return result

def create_dataframe(path):
    data = []
    for _, _, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                # Truncate file extension and split filename identifiers
                identifiers = file[:-len(".wav")].split('-')
                # Append file path w.r.t to root directory
                identifiers.append(os.path.join('Actor_' + identifiers[-1], file))
                # Append identifier to data
                data.append(identifiers)

    # Create pandas dataframe
    df = pd.DataFrame(data, columns=['modality', 'vocal_channel', 'emotion', 'intensity',
                                     'statement', 'repetition', 'actor', 'file'], dtype=np.float32)
    return df


if __name__ == '__main__':
    # Ignore warnings from librosa when extracting features
    warnings.filterwarnings('ignore')

    # Set the dataset path
    dataset_path = './ravdess/audio_speech_actors_01-24/'

    # Create a dataframe from the dataset
    audio_df = create_dataframe(dataset_path)

    # Set the feature extraction parameters
    cut_length = 2.5
    features_list = []
    emotions_list = []

    # Extracting features
    for idx in tqdm(range(len(audio_df))):
        # Load the audio file
        audio_path = os.path.join(dataset_path, audio_df.loc[idx, 'file'])
        waveform, rate = torchaudio.load(audio_path)
        waveform = waveform.mean(0)
        emotion = int(audio_df.loc[idx, 'emotion'])-1

        # Extract feature from the audio file
        window_size = int(cut_length * rate)
        sliding_size = window_size // 2
        start_idx, end_idx = 0, window_size

        # Waveform Length
        wavelen = len(waveform)
        while end_idx <= wavelen:
            feature = extract_features(waveform.numpy()[start_idx:end_idx], rate)
            features_list.append(feature)
            emotions_list.append(emotion)
            start_idx += sliding_size
            end_idx += sliding_size

        padded_waveform = np.pad(waveform.numpy()[start_idx:], 
                                 (0, window_size - (wavelen - start_idx)), 'constant')
        feature = extract_features(padded_waveform, rate)
        features_list.append(feature)
        emotions_list.append(emotion)

    # Save the features and labels
    result_features = pd.DataFrame(features_list)
    result_features['emotion'] = emotions_list
    result_features.to_csv('./features.csv', index=False)
    print('Features extracted and saved successfully')

    