import numpy as np
import torch
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from animals_dataset_v2 import AnimalDataset
from animals_models import CNN
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomAffine, ColorJitter
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, Adagrad, RMSprop
from sklearn.metrics import accuracy_score


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 50
    log_path = "tensorboard"
    transform = Compose([
        ToTensor(), # chuyen doi ve c x h x w
        Resize((224, 224)) # moi anh co 1 kich thuoc
    ])

    train_set = AnimalDataset(root='data/animals_v2/', train=True, transform=transform)
    training_params = {
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 6,
        "drop_last": True,
    }
    test_params = {
        "batch_size": 32,
        "shuffle": False,
        "num_workers": 6,
        "drop_last": False,
    }

    valid_set = AnimalDataset(root='data/animals_v2/', train=False, transform=transform)
    training_dataloader = DataLoader(train_set, **training_params)
    valid_dataloader = DataLoader(valid_set, **test_params)

    model = CNN(num_classes=len(train_set.categories)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.mkdir(log_path)
    writer = SummaryWriter(log_path)

    for epoch in range(num_epochs):
        model.train()
        losses = []
        # model.eval()
        for iter, (images, labels) in enumerate(training_dataloader):
            # Forward pass
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss_value = criterion(predictions, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            loss = loss_value.item()
            losses.append(loss)
            print("Epoch {}/{}. Iteration {}/{}. Loss value: {}".format(epoch+1, num_epochs, iter, len(training_dataloader), loss_value.item()))
            writer.add_scalar('Train/Loss', np.mean(losses), epoch*len(training_dataloader)+iter)

        model.eval()
        losses = []
        all_predictions = []
        all_gts = []
        with torch.no_grad(): # with torch.inference_mode():
            for iter, (images, labels) in enumerate(valid_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images)
                max_idx = torch.argmax(predictions, dim=1)
                # _, max_idx = torch.max(max_idx, dim=1)
                loss_value = criterion(predictions, labels)
                losses.append(loss_value.item())
                all_gts.extend(labels.tolist())
                all_predictions.extend(max_idx.tolist())

        writer.add_scalar('Valid/Loss', np.mean(losses), epoch)
        acc = accuracy_score(all_gts, all_predictions)
        writer.add_scalar('Valid/Accuracy', acc, epoch)
if __name__ == '__main__':
    train()
