import numpy as np
import torch
import os
import shutil

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from animals_dataset_v2 import AnimalDataset
from animals_models import CNN
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomAffine, ColorJitter
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, Adagrad, RMSprop
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import argparse
from tqdm.autonotebook import tqdm
import warnings
from PIL import Image

warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def get_args():
    parser = argparse.ArgumentParser(description='Animal classifier')
    parser.add_argument('-p', '--data_path', type=str, default="data/animals_v2")
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-l', '--lr', type=float, default=1e-2)  # SGD: lr = 1e-2. Adam: lr = 1e-3
    parser.add_argument('-s', '--image_size', type=int, default=224)
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None)
    parser.add_argument('-t', '--tensorboard_path', type=str, default="tensorboard")
    parser.add_argument('-r', '--trained_path', type=str, default="trained_models")
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = Compose([
        ToTensor(), # chuyen doi ve c x h x w
        Resize((args.image_size, args.image_size))
    ])

    train_set = AnimalDataset(root=args.data_path, train=True, transform=transform)
    training_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 6,
        "drop_last": True,
    }
    test_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 6,
        "drop_last": False,
    }

    valid_set = AnimalDataset(root=args.data_path, train=False, transform=transform)
    training_dataloader = DataLoader(train_set, **training_params)
    valid_dataloader = DataLoader(valid_set, **test_params)

    model = CNN(num_classes=len(train_set.categories)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = 0
    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.mkdir(args.tensorboard_path)
    if not os.path.isdir(args.trained_path):
        os.mkdir(args.trained_path)
    writer = SummaryWriter(args.tensorboard_path)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        # model.eval()
        progress_bar = tqdm(training_dataloader, colour="cyan")

        for iter, (images, labels) in enumerate(progress_bar):
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
            progress_bar.set_description("Epoch {}/{}. Loss value: {:.5f}".format(epoch+1, args.epochs, loss_value))

            # print("Epoch {}/{}. Iteration {}/{}. Loss value: {}".format(epoch+1, args.epochs, iter, len(training_dataloader), loss_value.item()))
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
        conf_matrix = confusion_matrix(all_gts, all_predictions)
        plot_confusion_matrix(writer, conf_matrix, [i for i in range(len(train_set.categories))], epoch=epoch)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "batch_size": args.batch_size,
        }
        torch.save(model.state_dict(), os.path.join(args.trained_path, 'last.pt'))
        if acc > best_acc:
            torch.save(model.state_dict(), os.path.join(args.trained_path, 'best.pt'))
            best_acc = acc
        scheduler.step()
if __name__ == '__main__':
    args = get_args()

    train(args)
