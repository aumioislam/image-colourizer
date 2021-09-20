import os
import cv2 as cv
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def param_init(l):
    if l.__class__.__name__ != 'BatchNorm2d':
        if hasattr(l, 'weight'):
            nn.init.xavier_uniform_(l.weight)
        
        if hasattr(l, 'bias'):
            nn.init.zeros_(l.bias)

def load(model, training=True, opt=None, learning_rate=1e-3):
    if training:
        if os.path.isfile(model.train_path):
            checkpoint = torch.load(model.train_path);
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['opt_state_dict'])
            epoch = checkpoint['epoch']
            model.train()
            return model, opt, epoch
        else:
            model.apply(param_init)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
            return model, opt, 0 

    else:
        if os.path.isfile(model.inference_path):
            model = torch.load(model.inference_path)
            model.eval()
            return model
        else:
            return None

#save function should only be called after being trained
def save(model, opt, epoch):
    torch.save({
                'model_state_dict' : model.state_dict(),
                'opt_state_dict' : opt.state_dict(),
                'epoch' : epoch,
                }, model.train_path)

    torch.save(model.state_dict(), model.inference_path)
    return

class colorizers(nn.Module):
    def __init__(self, inference_path, train_path):
        super(colorizers, self).__init__()

        self.inference_path = inference_path
        self.train_path = train_path


class CNN(colorizers):
    def __init__(self, dims=(256,256)):
        cnn_inf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/CNN_Inference.pt')
        cnn_train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/CNN_Training.tar')
        super(CNN, self).__init__(cnn_inf_path, cnn_train_path)

        self.layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(8, 2, kernel_size=3, padding=1),
                nn.Upsample(size=dims, mode='bilinear', align_corners=True),
                nn.Tanh()
            )

    def forward(self, x):
        x = torch.div(x, 100)
        x = self.layers(x)
        return torch.mul(x, 128)

    def loss(self, loss_func, X, Y):
        return loss_func(self(X), Y);


class CNN_AE(colorizers):
    def __init__(self, in_channel=1,
                out_channel=2,
                kernel=5,
                stride=1,
                feature_dim=16*122*122,
                latent_space=256):
        cnn_ae_inf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/CNN_AE_Inference.pt')
        cnn_ae_train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/CNN_AE_Training.tar')
        super(CNN_AE, self).__init__(cnn_ae_inf_path, cnn_ae_train_path)

        #Encoder Layers
        self.encConv1 = nn.Conv2d(in_channel, 4, kernel_size=kernel, stride=stride, padding=2)
        self.encBN1 = nn.BatchNorm2d(4)

        self.encConv2 = nn.Conv2d(4, 8, kernel_size=kernel, stride=stride)
        self.encBN2 = nn.BatchNorm2d(8)

        self.encConv3 = nn.Conv2d(8, 16, kernel_size=3, stride=stride)
        self.encBN3 = nn.BatchNorm2d(16)

        self.encFlat = nn.Flatten()
        self.encFC = nn.Linear(feature_dim, latent_space)

        #Decoder Layers
        self.decFC = nn.Linear(latent_space, feature_dim)
        self.decConv1 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=stride)
        self.decConv2 = nn.ConvTranspose2d(8, 4, kernel_size=kernel, stride=stride)
        self.decConv3 = nn.ConvTranspose2d(4, out_channel, kernel_size=kernel, stride=stride, padding=2)

    def encoder(self, x):
        x = F.relu(self.encBN1(self.encConv1(x)))
        x = F.relu(self.encBN2(self.encConv2(x)))
        x = F.relu(self.encBN3(self.encConv3(x)))
        x = self.encFlat(x)
        x = F.relu(self.encFC(x))
        return x

    def decoder(self, z):
        z = F.relu(self.decFC(z))
        z = z.view(-1, 16, 122, 122)
        z = F.relu(self.decConv1(z))
        z = F.relu(self.decConv2(z))
        z = torch.sigmoid(self.decConv3(z))
        return z

    def forward(self, x):
        x = torch.div(x, 100);
        z = self.encoder(x)
        out = self.decoder(z)
        return torch.mul(out, 128);

    def loss(self, loss_func, X, Y):
        return loss_func(self(X), Y);

#TODO: Implement VAE
#Add forward and backward encoders
#Reparameterization trick
#KL divergence loss
class CNN_VAE(colorizers):
    def __init__(self, in_channel=1,
                out_channel=2,
                kernel=5,
                stride=1,
                feature_dim=32*116*116,
                latent_space=256):
        cnn_vae_inf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/CNN_VAE_Inference.pt')
        cnn_vae_train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/CNN_VAE_Training.tar')
        super(CNN_VAE, self).__init__()


    #implement the reparameterization trick 
    def reparameterize(self, x):
        return


    def forward(self, x):
        return out


    #implement reconstruction loss using Kullback-Leibler divergence term
    def loss(self):
        return 
