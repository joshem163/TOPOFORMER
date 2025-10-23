import networkx as nx
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
from data_loader import load_dataset_ricci_and_degcent,load_label, load_dataset_ricci_and_hks,load_dataset_degcent_and_hks
from models import DualTransformerClassifier, reset_weights
from logger import print_stat,stat
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def main():
    parser = argparse.ArgumentParser(description='experiment')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='COX2')#MUTAG,PROTEINS,BZR,IMDB-BINARY,COX2,IMDB-MULTI,REDDIT-BINARY,REDDIT-MULTI-5K
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--gap_pmeter', type=int, default=2)
    parser.add_argument('--head', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    #dataset='REDDIT-MULTI-5K'#MUTAG,PROTEINS,BRZ,IMDB-BINARY,COX2,IMDB-MULTI,REDDIT-BINARY,REDDIT-MULTI-5K
    from sklearn.model_selection import train_test_split
    print(f"Processing dataset: {args.dataset}")

    # Load features and labels
    # features1, features2 = load_dataset_degcent_and_hks(args.dataset, args.gap_pmeter)
    features1, features2 = load_dataset_ricci_and_hks(args.dataset, args.gap_pmeter)
    graph_label = load_label(args.dataset)

    # Convert to PyTorch tensors
    X1 = torch.tensor(features1, dtype=torch.float32)
    X2 = torch.tensor(features2, dtype=torch.float32)
    y = torch.tensor(graph_label, dtype=torch.long)

    # Extract dataset details
    num_samples = len(X1)
    num_timesteps = len(X1[0])
    num_features1 = len(X1[0][0])
    num_features2 = len(X2[0][0])
    num_classes = len(np.unique(y))

    # Define input and output dimensions
    hidden_dim = args.hidden_channels
    output_dim = num_classes
    n_heads = args.head
    n_layers = args.num_layers


    # K-Fold Cross Validation
    kfold = KFold(n_splits=args.runs, shuffle=True,random_state=123)
    acc_per_fold = []
    fold_no = 1

    for train_idx, test_idx in kfold.split(X1):
        # Split data
        X1_train, X1_test = X1[train_idx], X1[test_idx]
        X2_train, X2_test = X2[train_idx], X2[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create DataLoader
        train_data = TensorDataset(X1_train, X2_train, y_train)
        test_data = TensorDataset(X1_test, X2_test, y_test)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # Initialize model, loss function, and optimizer
        model = DualTransformerClassifier(num_features1, num_features2, hidden_dim, output_dim, n_heads, n_layers,
                                          num_timesteps,0.0)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Lists to store metrics
        train_losses = []
        train_accuracies = []
        test_accuracies = []

        # Train the model
        reset_weights(model)
        #         optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
        model.train()
        # for epoch in tqdm(range(args.epochs), desc="Processing"):
        for epoch in range(args.epochs):
            epoch_train_loss = 0
            correct_train = 0
            total_train = 0

            for X1_batch, X2_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X1_batch, X2_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                # Track training loss and accuracy
                epoch_train_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total_train += y_batch.size(0)
                correct_train += (predicted == y_batch).sum().item()
            # print(f"Epoch {epoch+1}: alpha = {model.alpha.item():.4f}")

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_accuracy = correct_train / total_train
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

            # Evaluate the model
            model.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for X1_batch, X2_batch, y_batch in test_loader:
                    output = model(X1_batch, X2_batch)
                    _, predicted = torch.max(output, 1)
                    total_test += y_batch.size(0)
                    correct_test += (predicted == y_batch).sum().item()

            test_accuracy = correct_test / total_test
            test_accuracies.append(test_accuracy)

            model.train()
        print(f'Score for fold {fold_no}: ')
        acc = print_stat(train_accuracies, test_accuracies)
        acc_per_fold.append(acc)
        fold_no += 1
    stat(acc_per_fold, 'accuracy')
if __name__ == "__main__":
    main()