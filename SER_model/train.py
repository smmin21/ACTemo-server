from datasets.dataset import IemoCapDataset, RavdessDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from models.model import FeatureModel, IemoClassifier, RavClassifier
import torch.nn.functional as F
from tqdm import tqdm

if __name__ == "__main__":
    # You should define your own Feature CSV file path
    data_type = "iemocap"
    if data_type == "iemocap":
        dataset = IemoCapDataset("./datasets/iemocap_features.csv")
    elif data_type == "ravdess":
        dataset = RavdessDataset("./datasets/ravdess_features.csv")
    else:
        raise ValueError("Invalid data type")

    # Split the dataset into training and validation
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-int(len(dataset)*0.2), int(len(dataset)*0.2)])

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Define the model and optimizer, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = IemoClassifier(num_classes=7) if data_type == "iemocap" else RavClassifier(num_classes=8)
    model = nn.Sequential(FeatureModel(), classifier).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train/Val Accuracy Logging
    best_train_acc = {"epoch": 0, "acc": 0}
    best_val_acc = {"epoch": 0, "acc": 0}

    # Training Loop
    for epoch in range(1000):
        # Average Loss and Accuracy Logging
        running_loss = 0
        correct, total = 0, 0

        # Progress Bar with tqdm
        progress_bar = tqdm(enumerate(train_dataloader))

        # Set model into training mode
        model.train()
        for iteration, data in progress_bar:
            # Extract feature and emotion from the dataloader
            feature, emotion = data
            feature, emotion = feature.to(device), emotion.to(device)

            # Forward Pass
            # Unsqueeze the feature to add channel dimension (B, L) -> (B, 1, L)
            pred = model(feature.unsqueeze(1))

            # Compute Loss and Backward Pass
            optimizer.zero_grad()
            loss = F.cross_entropy(pred, emotion)
            loss.backward()
            optimizer.step()

            # Logging
            running_loss += loss.item()
            correct += (pred.argmax(1) == emotion).sum().item()
            total += emotion.size(0)

            # Update Progress Bar
            progress_bar.set_description(f"Epoch {epoch}, Iteration {iteration}, Loss: {running_loss/(iteration+1):.4f}, Acc: {(100*correct/total):.2f}%")

        # Logging the best train accuracy
        if (100*correct/total) > best_train_acc["acc"]:
            best_train_acc["epoch"] = epoch
            best_train_acc["acc"] = 100*correct/total

        # Validation Loop
        # Set model into evaluation mode
        model.eval()
        with torch.no_grad():
            # Average Loss and Accuracy Logging
            val_loss = 0.0
            val_correct, val_total = 0, 0
            for iteration, data in enumerate(val_dataloader):
                feature, emotion = data
                feature, emotion = feature.to(device), emotion.to(device)
                pred = model(feature.unsqueeze(1))
                loss = nn.CrossEntropyLoss()(pred, emotion)
                val_loss += loss.item()
                val_correct += (pred.argmax(1) == emotion).sum().item()
                val_total += emotion.size(0)
                
            print(f"[Epoch {epoch+1}/{1000}] Validation Loss: {val_loss/(iteration+1):.4f}, Validation Acc: {(100*val_correct/val_total):.2f}%")

            if (100*val_correct/val_total) > best_val_acc["acc"]:
                best_val_acc["epoch"] = epoch
                best_val_acc["acc"] = 100*val_correct/val_total
                # Save the best model
                torch.save(model.state_dict(), f"best_model_{data_type}.pth")

    # Save the last model
    torch.save(model.state_dict(), f"last_model_{data_type}.pth")
    print("=========================")
    print(f"Best Train Acc: {best_train_acc['acc']:.2f}% at epoch {best_train_acc['epoch']}")
    print(f"Best Val Acc: {best_val_acc['acc']:.2f}% at epoch {best_val_acc['epoch']}")