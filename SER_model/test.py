import torchaudio
import torch.nn as nn
from datasets.feature_extraction import extract_features, extract_features_parallel
from models.model import FeatureModel, IemoClassifier, RavClassifier
import numpy as np
import torch
import pdb
from time import time
import multiprocessing
import threading
import logging

logging.basicConfig(filename='test.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)

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
    lock = threading.Lock()

    def thread_task(chunk):
        feature = extract_features(chunk, rate)
        with lock:
            features_list.append(feature)

    for chunk in waveform_chunks:
        thread = threading.Thread(target=thread_task, args=(chunk,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    return features_list

def run_test(data_path, emotion='Angry'):
    # Logging
    logging.info("=====================================")
    logging.info(f"Test for [{data_path}] with emotion [{emotion}] started")
    time_per_data = time()

    # Custom Input
    train_data_type = 'iemocap'
    emotion2idx = {'Angry': 0, 'Distressed': 0, 
                   'Content': 1, 'Happy': 1, 
                   'Excited': 2, 'Joyous': 2, 
                   'Depressed': 3, 'Sad': 3, 
                   'Bored': 4, 'Calm': 4, 'Relaxed': 4, 'Sleepy': 4, 
                   'Afraid': 5, 'Surprised': 6}
    emotions_list = ['ang', 'hap', 'exc', 'sad', 'neu', 'fea', 'sur']
    model_path = './models/model_finetuned.pth'
    parallel = 'threading'

    # Extract features from the audio
    feature_extraction_start = time()
    if parallel:
        features_list = extract_features_from_audio_parallel(data_path, parallel)
    else:
        features_list = extract_features_from_audio(data_path)
    feature = torch.tensor(np.array(features_list), dtype=torch.float32)
    print(f"Feature extraction: {time()-feature_extraction_start:.2f} seconds")
    logging.info(f"Feature extraction: {time()-feature_extraction_start:.2f} seconds")

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

        # 1. Get the label emotion statistics
        pred = nn.functional.softmax(pred, dim=1)
        specific_emotion_score = pred[:, emotion2idx[emotion]]
        sort_specific_emotion_score = torch.sort(specific_emotion_score, descending=True)
        print(f"Specific emotion score: {specific_emotion_score}")
        print(f"Mean Specific emotion score: {specific_emotion_score.mean():.2f}")
        print(f"Mean Specific emotion score except edge: {sort_specific_emotion_score.values[1:-1].mean():.2f}")

        # 2. Get the most profitable prediction (general)
        emotion_pred = torch.argmax(pred, dim=1)

        # 3. Get emotion rank and evaluate the level
        top_emotions = torch.argsort(pred, dim=1, descending=True)[:, :2]  # Top 2 emotions
        top_emotions_count = torch.bincount(top_emotions.flatten(), minlength=len(emotions_list))
        zero_indices = torch.nonzero(top_emotions_count == 0).flatten()
        sorted_indices = torch.argsort(top_emotions_count, descending=True)
        sorted_emotions = [emotions_list[idx] for idx in sorted_indices]
        print(f"Top emotions count: {top_emotions_count}")
        print(f"Frequency rank of top emotions: {sorted_emotions}")
        # [level] A: 1st, B: 2nd ~ 4th, C: 5th ~ 7th
        level_idx = sorted_emotions.index(emotions_list[emotion2idx[emotion]])
        level = 'A' if level_idx == 0 else 'B' if 1 <= level_idx <= 3 else 'C'
        level = 'C' if emotion2idx[emotion] in zero_indices else level
        print(f"Level of prediction: {level}")
        
    print(f"Time taken for inference: {time()-time_per_data:.2f} seconds")
    logging.info(f"Time taken for inference: {time()-time_per_data:.2f} seconds")
    # predicted emotion, level
    return emotions_list[emotion_pred.mode().values.item()], level

if __name__ == '__main__':
    # Logging 
    logging.info("")
    logging.info("=====================================")
    logging.info("========== Test Started =============")

    # while True:
    #     # Custom Input
    #     select_emotion = input("Enter the emotion: ")
    #     data_path = input("Enter the path of the audio file: ")
    #     predicted_emotion, level = run_test(data_path, select_emotion)
    #     print(f'[FINAL] Predicted emotion : {predicted_emotion}, Level: {level}')

    print("Angry 1 Test")
    run_test('test_data/angry1.wav', 'Angry')
    print("Angry 2 Test")
    run_test('test_data/angry1.wav', 'Angry')
    print("Angry 3 Test")
    run_test('test_data/angry1.wav', 'Angry')
    print("Afraid 1 Test")
    run_test('test_data/afraid1.wav', 'Afraid')
    print("Afraid 2 Test")
    run_test('test_data/afraid1.wav', 'Afraid')

       