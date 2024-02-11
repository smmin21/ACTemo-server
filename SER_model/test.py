import torchaudio
import torch.nn as nn
from datasets.feature_extraction import extract_features, extract_features_parallel
from models.model import FeatureModel, IemoClassifier, RavClassifier
import pandas as pd
import numpy as np
import torch
import pdb
from time import time
import multiprocessing
import threading


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
    return features_list

def extract_features_from_audio_parallel(data_path, parallel='threading'):
    # Load the audio file
    waveform, rate = torchaudio.load(data_path)
    waveform = waveform.mean(0)

    # Set the feature extraction parameters
    cut_length = 2.5
    features_list = []

    # Split waveform into chunks
    window_size = int(cut_length * rate)
    sliding_size = window_size // 2
    start_idx, end_idx = 0, window_size
    waveform_chunks = []
    while end_idx <= len(waveform):
        waveform_chunks.append(waveform.numpy()[start_idx:end_idx])
        start_idx += sliding_size
        end_idx += sliding_size
    # Pad the last chunk if needed
    if start_idx < len(waveform):
        padded_waveform = np.pad(waveform.numpy()[start_idx:],
                                 (0, window_size - (len(waveform) - start_idx)), 'constant')
        waveform_chunks.append(padded_waveform)

    # Extract features from the chunks
    parallel_type = {'process': multiprocessing_method1, 
                     'pool': multiprocessing_method2, 
                     'threading': multithreading_method}
    return parallel_type[parallel](waveform_chunks, rate, features_list)

def multiprocessing_method1(waveform_chunks, rate, features_list):
    # Method 1 : Using multiprocessing Process
    # Create a queue to store the features
    q = multiprocessing.Queue()
    # Create and start a process for each chunk
    processes = []
    for chunk in waveform_chunks:
        process = multiprocessing.Process(target=extract_features_parallel, args=(chunk, rate, q))
        processes.append(process)
        process.start()
    # Wait for all processes to finish
    for process in processes:
        process.join()
    # Get the features from the queue
    while not q.empty():
        features_list.append(q.get())
    return features_list

def multiprocessing_method2(waveform_chunks, rate, features_list):
    # Method 2 : Using multiprocessing Pool
    with multiprocessing.Pool() as pool:
        # get the num of worker processes
        num_workers = pool._processes
        print(f"Number of worker processes: {num_workers}")
        # extract features from the chunks
        features_list = pool.starmap(extract_features, [(chunk, rate) for chunk in waveform_chunks])
    return features_list

def multithreading_method(waveform_chunks, rate, features_list):
    # Method 3 : Using threading
    threads = []
    for chunk in waveform_chunks:
        thread = threading.Thread(target=lambda x: features_list.append(extract_features(x, rate)), args=(chunk,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    return features_list


if __name__ == '__main__':
    while True:
        # Custom Input
        data_path = input("Enter the path of the audio file: ")
        time_per_data = time()
        # data_path = './test_data/calmed1.wav'
        train_data_type = 'iemocap'
        emotions_list = ['ang', 'hap', 'exc', 'sad', 'neu', 'fea', 'sur']
        model_path = './test_data/best_model_iemocap.pth'
        parallel = 'threading'

        # Extract features from the audio
        feature_extraction_start = time()
        if parallel:
            features_list = extract_features_from_audio_parallel(data_path, parallel)
        else:
            features_list = extract_features_from_audio(data_path)
        feature = torch.tensor(np.array(features_list), dtype=torch.float32)
        print(f"Feature extraction: {time()-feature_extraction_start:.2f} seconds")

        # Define the model, device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier = IemoClassifier(num_classes=7) if train_data_type == "iemocap" else RavClassifier(num_classes=8)
        model = nn.Sequential(FeatureModel(), classifier).to(device)

        # Load the model
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Inference
        model.eval()
        with torch.no_grad():
            pred = model(feature.unsqueeze(1).to(device)) # input dim : (batch, channel, length)
            emotion_pred = torch.argmax(pred, dim=1)
            for idx, pred in enumerate(emotion_pred):
                print(f"Prediction for {idx+1}th segment: {pred.item()}")

        print(f"Time taken for inference: {time()-time_per_data:.2f} seconds")


