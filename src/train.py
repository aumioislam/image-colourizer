import os
import sys
import torch
import argparse
from colorizers import CNN, CNN_AE, param_init, load, save
from mirflickr_loader import mirflickrDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CNN', help='choose model for training')
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.0002')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        sys.exit(0)

    if args.model == 'CNN':
        model = CNN().to(device)
        dims = (256,256)
    if args.model == 'CNN_AE':
        model = CNN_AE().to(device)
        dims = (128,128)
    else:
        sys.exit(0)

    if args.batch_size > 0:
        batch_size = args.batch_size
    else:
        sys.exit(0)

    if args.epochs > 0:
        num_epochs = args.epochs
    else:
        sys.exit(0)

    if args.lr < 0.5:
        learning_rate = args.lr
    else:
        sys.exit(0)

    img_ds = mirflickrDataset(set_path='../mirflickr/train', dims=dims)
    train_dl = DataLoader(img_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, optimizer, epoch = load(model, opt=optimizer, learning_rate=learning_rate)

    for e in range(num_epochs):
        running_loss = 0

        if not torch.cuda.is_available():
            save(model, opt, epoch+e)

        for i, (x, y) in enumerate(train_dl):
            X = x.to(device)
            Y = y.to(device)
            loss = model.loss(F.mse_loss, X, Y)
            running_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch {}: Loss: {}'.format(e + epoch, running_loss))
    
    save(model, optimizer, epoch+num_epochs)
