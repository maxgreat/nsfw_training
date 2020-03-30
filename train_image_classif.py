import sys
import time

import torch
from absl import flags
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

FLAGS = flags.FLAGS
flags.DEFINE_string("name", None, "Name")
flags.DEFINE_string(
    "model", "resnet50", "Model name. Must be : resnet18/50/101 or resnext"
)
flags.DEFINE_string("data", "/data/porn/binary/", "Data path")
flags.DEFINE_integer("train_batch_size", 128, "Train Batch size")
flags.DEFINE_integer("val_batch_size", 1024, "Val Batch size")
flags.DEFINE_integer("workers", 8, "Number of workers to load data")
flags.DEFINE_float("lr", 1e-4, "Initial learning rate")
flags.DEFINE_integer("device", 1, "Device ID")
flags.DEFINE_string("tensorboard", "/data/porn/tensorboard/", "Tensoard logdir")


# Configuration Variables
MEAN_PREPRO = [0.485, 0.456, 0.406]
STD_PREPRO = [0.229, 0.224, 0.225]
RESIZE_PREPRO = 256, 256
DATA_DIR = FLAGS.data
TRAINSET_ROOT = f"{DATA_DIR}/train"
TESTSET_ROOT = f"{DATA_DIR}/test"
TRAIN_BATCH_SIZE = FLAGS.train_batch_size
TRAIN_SHUFFLE = True
TRAIN_NUM_WORKERS = FLAGS.workers
TRAIN_PIN_MEMORY = True

VAL_BATCH_SIZE = FLAGS.val_batch_size
VAL_SHUFFLE = False
VAL_NUM_WORKERS = FLAGS.workers
VAL_PIN_MEMORY = True

INITIAL_LR = FLAGS.lr
DEVICE_ID = FLAGS.device

TENSORBOARD_DIR = f"{FLAGS.tensorboard}{FLAGS.name}"
logger = SummaryWriter(TENSORBOARD_DIR)

###################################################################################
#
# TRAIN and VALIDATION LOOPS
#
###################################################################################
def train(train_loader, model, criterion, optimizer, epoch, on_iteration=None):
    model = model.train()
    print("Start Training")
    avg_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        print(f"{i/len(train_loader) * 100 : 2.2f}%", end="\r")
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
        if on_iteration is not None:
            on_iteration(
                iteration=i + epoch * len(train_loader),
                loss=loss,
                y_pred=outputs,
                y_true=labels,
            )
    return avg_loss / len(train_loader)


def validate(val_loader, model, criterion, print_freq=1000):
    model = model.eval()
    y_pred, y_true = [], []
    avg_loss = 0
    for i, (inputs, labels) in enumerate(val_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            outputs = model(inputs)
            avg_loss += criterion(outputs, labels).item()
        _, indices = torch.max(outputs.cpu(), 1)
        y_pred.append(indices)
        y_true.append(labels.cpu().clone())
    return avg_loss / len(val_loader), torch.cat(y_pred), torch.cat(y_true)


###################################################################################

###################################################################################
#
# UTILS
#
###################################################################################
def plot_cm(classes, cm):
    fig, ax = plt.subplots(figsize=(len(classes), len(classes)))
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt="d")
    # labels, title and ticks
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(classes, rotation=90)
    ax.yaxis.set_ticklabels(classes, rotation=0)
    return fig


def compute_porn_detection(epoch, y_pred, y_true):
    p_pred = (y_pred == 1) | (y_pred == 3)
    p_truth = (y_true == 1) | (y_true == 3)
    logger.add_pr_curve("Eval/Prec_recall", p_truth, p_pred, epoch)


def validation_logs(epoch, loss_avg, y_pred, y_true):
    logger.add_scalar("Loss/Avg_Val", loss_avg, epoch)
    logger.add_scalar(
        "Eval/Precision", (y_pred == y_true).sum() / float(len(y_pred)), epoch
    )
    compute_porn_detection(epoch, y_pred, y_true)
    cm = confusion_matrix(y_pred, y_true)
    cm_image = plot_cm(train_dataset.classes, cm)
    logger.add_figure("Eval/Confusion Matrix", cm_image, epoch)


def on_iteration_logs(iteration, loss, y_pred, y_true):
    loss_value = loss.item()
    if iteration % 50 == 0:
        logger.add_scalar("Loss/Train", loss_value, iteration)
        print(f"{iteration}/{len(train_loader)} \t" f"Loss {loss_value}")


###################################################################################


normalize = transforms.Normalize(mean=MEAN_PREPRO, std=STD_PREPRO)
prepro = transforms.Compose(
    [
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)
prepro_val = transforms.Compose(
    [transforms.Resize(RESIZE_PREPRO), transforms.ToTensor(), normalize]
)


train_dataset = ImageFolder(TRAINSET_ROOT, transform=prepro)
val_dataset = ImageFolder(TESTSET_ROOT, transform=prepro_val)

print(train_dataset)
print(train_dataset.classes)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=TRAIN_SHUFFLE,
    num_workers=TRAIN_NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=VAL_SHUFFLE,
    num_workers=VAL_NUM_WORKERS,
    pin_memory=VAL_PIN_MEMORY,
    drop_last=False,
)

if FLAGS.model == "resnet18":
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, len(train_dataset.classes))
elif FLAGS.model == "resnet50":
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, len(train_dataset.classes))
elif FLAGS.model == "resnext":
    model = models.resnext101_32x8d(pretrained=True)
    model.fc = torch.nn.Linear(2048, len(train_dataset.classes))
else:
    print("Unknow model")
    sys.exit(1)

DEVICE = f"cuda:{DEVICE_ID}"
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()

# First train the Fully Connected layer
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=INITIAL_LR
)
for i in range(10):
    val_loss, y_pred, y_true = validate(val_loader, model, criterion)
    validation_logs(i, val_loss, y_pred, y_true)
    loss = train(
        train_loader, model, criterion, optimizer, i, on_iteration=on_iteration_logs
    )
    logger.add_scalar("Loss/Avg_Train", loss, i)


# Now train the second part of the network
for i, child in enumerate(model.children()):
    if i > 6:
        for p in child.parameters():
            p.requires_grad = True
train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE // 2,  # to fit memory
    shuffle=TRAIN_SHUFFLE,
    num_workers=TRAIN_NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=INITIAL_LR * 0.1,  # divide learning by 10
)

for i in range(10, 40):
    val_loss, y_pred, y_true = validate(val_loader, model, criterion)
    validation_logs(i, val_loss, y_pred, y_true)
    loss = train(
        train_loader, model, criterion, optimizer, i, on_iteration=on_iteration_logs
    )
    logger.add_scalar("Loss/Avg_Train", loss, i)


# Now train almost all the network (exept the first 3 children)
for i, child in enumerate(model.children()):
    if i > 3:
        for p in child.parameters():
            p.requires_grad = True
train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE // 2,  # to fit memory
    shuffle=TRAIN_SHUFFLE,
    num_workers=TRAIN_NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=INITIAL_LR * 0.01,  # divide learning by 10
)

for i in range(40, 60):
    val_loss, y_pred, y_true = validate(val_loader, model, criterion)
    validation_logs(i, val_loss, y_pred, y_true)
    loss = train(
        train_loader, model, criterion, optimizer, i, on_iteration=on_iteration_logs
    )
    logger.add_scalar("Loss/Avg_Train", loss, i)
