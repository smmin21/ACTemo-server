import torchaudio
import torch.nn as nn
from datasets.feature_extraction import extract_features
from models.model import FeatureModel, IemoClassifier, RavClassifier
import pandas as pd
import numpy as np
import torch
import pdb
from time import time


def extract_features_from_audio(data_path):
    # Load the audio file
    waveform, rate = torchaudio.load(data_path)
    waveform = waveform.mean(0)

    # Set the feature extraction parameters
    cut_length = 2.5
    features_list = []

    # Extract feature from the audio file
    window_size = int(cut_length * rate)
    sliding_size = window_size // 2
    start_idx, end_idx = 0, window_size

    # Waveform Length
    wavelen = len(waveform)
    while end_idx <= wavelen:
        feature = extract_features(waveform.numpy()[start_idx:end_idx], rate)
        features_list.append(feature)
        start_idx += sliding_size
        end_idx += sliding_size

    padded_waveform = np.pad(waveform.numpy()[start_idx:],
                             (0, window_size - (wavelen - start_idx)), 'constant')
    feature = extract_features(padded_waveform, rate)
    features_list.append(feature)

    result_features = pd.DataFrame(features_list)
    return result_features


if __name__ == '__main__':
    # Custom Input
    while True:
        data_path = input("Enter the path of the audio file: ")
        time_per_data = time()
        # data_path = './test_data/calmed1.wav'
        train_data_type = 'iemocap'
        model_path = './test_data/best_model_iemocap.pth'

        # Extract features from the audio
        feature = torch.tensor(extract_features_from_audio(data_path).iloc[0].values, dtype=torch.float32)

        # Define the model, device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier = IemoClassifier(num_classes=7) if train_data_type == "iemocap" else RavClassifier(num_classes=8)
        model = nn.Sequential(FeatureModel(), classifier).to(device)

        # Load the model
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Inference
        model.eval()
        with torch.no_grad():
            pred = model(feature.unsqueeze(0).unsqueeze(0).to(device)) # input dim : (batch, channel, length)
            print(pred.argmax(1).item(), pred)

        print(f"Time taken for inference: {time()-time_per_data:.2f} seconds")


