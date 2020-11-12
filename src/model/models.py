import torch
import torch.nn as nn
import torch.nn.functional as F

class FxNet(nn.Module):
    
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes # number of fx labels

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*29*18, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=self.n_classes)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
<<<<<<< HEAD
=======
        t = self.batchNorm1(t)
>>>>>>> fxnet_bn
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
<<<<<<< HEAD
=======
        t = self.batchNorm2(t)
>>>>>>> fxnet_bn
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden dense layer
        t = t.reshape(-1, 12*29*18)
        t = self.fc1(t)
<<<<<<< HEAD
=======
        t = self.batchNorm3(t)
>>>>>>> fxnet_bn
        t = F.relu(t)
        
        # (5) hidden dense layer
        t = self.fc2(t)
<<<<<<< HEAD
=======
        t = self.batchNorm4(t)
>>>>>>> fxnet_bn
        t = F.relu(t)
        
        # (6) output dense layer
        t = self.out(t)

        return t


class SettingsNet(nn.Module):
    
    def __init__(self, n_settings):
        super().__init__()
        self.n_settings = n_settings # number of settings

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.batchNorm1 = nn.BatchNorm2d(num_features=6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.batchNorm2 = nn.BatchNorm2d(num_features=12)

        self.fc1 = nn.Linear(in_features=12*29*18, out_features=120)
        self.batchNorm3 = nn.BatchNorm1d(num_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.batchNorm4 = nn.BatchNorm1d(num_features=60)
        self.out = nn.Linear(in_features=60, out_features=self.n_settings)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = self.batchNorm1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = self.batchNorm2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden dense layer
        t = t.reshape(-1, 12*29*18)
        t = self.fc1(t)
        t = self.batchNorm3(t)
        t = F.relu(t)

        # (5) hidden dense layer
        t = self.fc2(t)
        t = self.batchNorm4(t)
        t = F.relu(t)

        # (6) output dense layer
        t = self.out(t)
        t = F.tanh(t)

        return t


class MultiNet(nn.Module):
    
    def __init__(self, n_classes, n_settings):
        super().__init__()
        self.n_classes = n_classes
        self.n_settings = n_settings

        # common network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.batchNorm1 = nn.BatchNorm2d(num_features=6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.batchNorm2 = nn.BatchNorm2d(num_features=12)
        # fx network
        self.fc1_a = nn.Linear(in_features=12*29*18, out_features=120)
        self.batchNorm3_a = nn.BatchNorm1d(num_features=120)
        self.fc2_a = nn.Linear(in_features=120, out_features=60)
        self.batchNorm4_a = nn.BatchNorm1d(num_features=60)
        self.out_a = nn.Linear(in_features=60, out_features=self.n_classes)
        # settings network
        self.fc1_b = nn.Linear(in_features=12*29*18, out_features=120)
        self.batchNorm3_b = nn.BatchNorm1d(num_features=120)
        self.fc2_b = nn.Linear(in_features=120, out_features=60)
        self.batchNorm4_b = nn.BatchNorm1d(num_features=60)
        self.out_b = nn.Linear(in_features=60, out_features=self.n_settings)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) common conv layer
        t = self.conv1(t)
        t = self.batchNorm1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) common conv layer
        t = self.conv2(t)
        t = self.batchNorm2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = t.reshape(-1, 12*29*18)
        
        # (4a) fx dense layer
        t_a = self.fc1_a(t)
        t_a = self.batchNorm3_a(t_a)
        t_a = F.relu(t_a)

        # (5a) fx dense layer
        t_a = self.fc2_a(t_a)
        t_a = self.batchNorm4_a(t_a)
        t_a = F.relu(t_a)

        # (6a) fx output layer
        t_a = self.out_a(t_a)

        # (4b) set dense layer
        t_b = self.fc1_b(t)
        t_b = self.batchNorm3_b(t_b)
        t_b = F.relu(t_b)

        # (5b) set dense layer
        t_b = self.fc2_b(t_b)
        t_b = self.batchNorm4_b(t_b)
        t_b = F.relu(t_b)

        # (6b) set output layer
        t_b = self.out_b(t_b)
        t_b = F.tanh(t_b)
        
        return t_a, t_b
    
    
class SettingsNetCond(nn.Module):

    def __init__(self, n_settings, mel_shape, num_embeddings, embedding_dim=50):
        super().__init__()
        self.n_settings = n_settings
        self.mel_shape = mel_shape
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.fc0 = nn.Linear(in_features=embedding_dim, out_features=self.mel_shape[1] * self.mel_shape[2])

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=6, kernel_size=5)
        self.batchNorm1 = nn.BatchNorm2d(num_features=6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.batchNorm2 = nn.BatchNorm2d(num_features=12)

        self.fc1 = nn.Linear(in_features=12*29*18, out_features=120)
        self.batchNorm3 = nn.BatchNorm1d(num_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.batchNorm4 = nn.BatchNorm1d(num_features=60)
        self.out = nn.Linear(in_features=60, out_features=self.n_settings)

    def forward(self, t, c):
        # (0.1) embedding layer
        c = self.emb(c)
        # (0.2) dense
        c = self.fc0(c)
        # transform
        c = c.reshape(-1, 1, self.mel_shape[1], self.mel_shape[2])

        # (1) input layer
        t = torch.cat((t, c), dim=1)

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden dense layer
        t = t.reshape(-1, 12*29*18)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden dense layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output dense layer
        t = self.out(t)
        t = F.tanh(t)

        return t