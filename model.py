import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TrainModel(nn.Module):
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim, device):
        super(TrainModel, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._device = device
        self._model = self._build_model(num_layers, width, input_dim, output_dim)
        self._optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)
        self._loss = nn.MSELoss()

    def _build_model(self, num_layers, width, input_dim, output_dim):
        """
        Build a fully connected deep neural network
        """
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(num_layers):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, output_dim)]
        model = nn.Sequential(*layers)
        return model

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            return self._model(state)

    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        states = torch.tensor(states, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            return self._model(states)

    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        states = torch.tensor(states, dtype=torch.float32).to(self._device)
        q_sa = torch.tensor(q_sa, dtype=torch.float32).to(self._device)
        self._optimizer.zero_grad()
        predictions = self._model(states)
        loss = self._loss(predictions, q_sa)
        loss.backward()
        self._optimizer.step()

    def save_model(self, path):
        """
        Save the current model in the folder as a .pth file
        """
        torch.save(self._model.state_dict(), os.path.join(path, 'trained_model.pth'))

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size


class TestModel(nn.Module):
    def __init__(self, input_dim, width, num_layers, output_dim, model_path, device):
        super(TestModel, self).__init__()
        self._input_dim = input_dim
        self._device = device
        self._model = self._build_model(num_layers, width, input_dim, output_dim)
        self._load_my_model(model_path)

    def _build_model(self, num_layers, width, input_dim, output_dim):
        """
        Build a fully connected deep neural network
        """
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(num_layers):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, output_dim)]
        model = nn.Sequential(*layers)
        return model

    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.pth')
        if os.path.isfile(model_file_path):
            self._model.load_state_dict(torch.load(model_file_path))
            self._model.to(self._device)
            self._model.eval()
        else:
            sys.exit("Model number not found")

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            return self._model(state)

    @property
    def input_dim(self):
        return self._input_dim
