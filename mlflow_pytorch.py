# %% [markdown]
# # Hyperparameters Tuning using Optuna and MLflow Tracking

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import kagglehub
import random
import matplotlib.pyplot as plt
import optuna
import mlflow
import mlflow.pytorch

# %%
# Setup device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Set MLflow experiment
mlflow.set_experiment("FashionMNIST_Optimization")

# %% [markdown]
# ## Loading and Preparing Dataset

# %%
dataset_path = kagglehub.dataset_download('zalando-research/fashionmnist')
df_train = pd.read_csv(f'{dataset_path}/fashion-mnist_train.csv')
df_test = pd.read_csv(f'{dataset_path}/fashion-mnist_test.csv')

X_train = df_train.iloc[:, 1:].values / 255.0
y_train = df_train.iloc[:, 0].values
X_test = df_test.iloc[:, 1:].values / 255.0
y_test = df_test.iloc[:, 0].values

# %%
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

train_dataset = CustomDataset(features=X_train, labels=y_train)
test_dataset = CustomDataset(features=X_test, labels=y_test)

# %% [markdown]
# ## Model Architecture

# %%
class Model(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, num_neurons):
        super().__init__()
        layers = []
        curr_input = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(curr_input, num_neurons))
            layers.append(nn.BatchNorm1d(num_neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            curr_input = num_neurons
        layers.append(nn.Linear(curr_input, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, features):
        return self.model(features)

# %% [markdown]
# ## Objective Function with MLflow Tracking

# %%
def objective(trial):
    # Suggest hyperparameters
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 3)
    num_neurons = trial.suggest_int('num_neurons', 32, 128)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 5, 20, step=5) # Reduced for demonstration speed
    batch_size = trial.suggest_categorical('batch_size', [64, 128])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])

    # Start a nested MLflow run for each Optuna trial
    with mlflow.start_run(nested=True):
        # Log trial hyperparameters to MLflow
        mlflow.log_params(trial.params)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = Model(X_train.shape[1], 10, num_hidden_layers, num_neurons).to(device)
        loss_fn = nn.CrossEntropyLoss()
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Training Loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch_feat, batch_lab in train_dataloader:
                batch_feat, batch_lab = batch_feat.to(device), batch_lab.to(device)
                
                optimizer.zero_grad()
                preds = model(batch_feat)
                loss = loss_fn(preds, batch_lab)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # Log epoch loss to MLflow
            mlflow.log_metric("train_loss", running_loss/len(train_dataloader), step=epoch)

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_feat, batch_lab in test_dataloader:
                batch_feat, batch_lab = batch_feat.to(device), batch_lab.to(device)
                outputs = model(batch_feat)
                _, predicted = torch.max(outputs, 1)
                total += batch_lab.size(0)
                correct += (predicted == batch_lab).sum().item()

        accuracy = (correct / total) * 100
        # Log final accuracy
        mlflow.log_metric("accuracy", accuracy)
        
        return accuracy

# %% [markdown]
# ## Execute Study

# %%
# Wrap study in a parent MLflow run for organization
with mlflow.start_run(run_name="Optuna_Main_Session"):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)

    # Log best results to parent run
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_accuracy", study.best_value)

    print("Optimization complete.")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")