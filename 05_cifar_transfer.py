# Quick run note:
# - Before running this file, activate your virtual environment using the pytorch bash alias:
#     pytorch
#   This ensures the correct Python, CUDA and PyTorch versions are available.

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import amp
from pathlib import Path

# Choose device: GPU if available, otherwise CPU.
# - 'cuda' means use NVIDIA GPU, which is much faster for training deep nets.
# - This line picks the best device automatically.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)


# Set the data directory
base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
data_dir = base_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# Data transforms:
# - We resize/crop and flip images randomly for training (this helps the model generalize).
# - We convert to tensors and normalize so pixel values are centered (helps learning).
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # small random crops -> tiny shifts in images
    transforms.RandomHorizontalFlip(),         # random flips -> more varied data
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),  # standard CIFAR stats
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
])

# Datasets and loaders:
# - CIFAR10 is a small image dataset (32x32 RGB) with 10 classes.
# - DataLoader gives us batches; shuffle=True for training so batches vary each epoch.

train_ds = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=transform_train)
test_ds  = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# Model: Pretrained ResNet18 with a new final layer
# - We load a ResNet18 (a common image model) with pretrained weights.
# - Transfer learning: freeze the backbone and replace the final classification head.
#   This lets us use features learned from a large dataset and only train a small new part.
weights = models.ResNet18_Weights.DEFAULT
net = models.resnet18(weights=weights)
for p in net.parameters():  # freeze backbone so its weights don't change during training
    p.requires_grad = False
num_feats = net.fc.in_features
net.fc = nn.Linear(num_feats, 10)  # new classifier for 10 CIFAR classes
net = net.to(device)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)  # use multiple GPUs automatically if available

# Optimizer: only update the new head (the rest is frozen)
# If the model was wrapped in DataParallel, the real module is at net.module.
# Use that when available so we can access .fc safely.
base_model = net.module if hasattr(net, "module") else net
opt = torch.optim.AdamW(base_model.fc.parameters(), lr=3e-3)  # only training head
# GradScaler and autocast help with mixed-precision training on GPUs:
# - use the torch.amp API; construct the scaler with the enabled flag only
scaler = amp.GradScaler(enabled=(device=='cuda'))

def run_epoch(loader, train=True):
    net.train(train)  # sets the model to train or eval mode (affects dropout/batchnorm)
    total, correct, running_loss, n = 0, 0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        # autocast: enables mixed-precision on GPU for speed (use torch.amp API)
        with amp.autocast('cuda', enabled=(device=='cuda')):
            logits = net(xb)                     # forward pass: network predictions (raw scores)
            loss = F.cross_entropy(logits, yb)   # loss: how wrong the model is (lower is better)
        if train:
            opt.zero_grad(set_to_none=True)           # clear gradients from previous step
            scaler.scale(loss).backward()             # scale and backpropagate gradients
            scaler.step(opt)                          # apply optimizer step (update weights)
            scaler.update()                           # update the scaler for next iteration
        running_loss += loss.item() * xb.size(0)
        pred = logits.argmax(1)                        # predicted class is the highest score
        correct += (pred == yb).sum().item()          # count correct predictions
        n += xb.size(0)
    return running_loss/n, correct/n  # return average loss and accuracy

# Training loop: run a few epochs and print train/test loss and accuracy
for epoch in range(1, 4):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    te_loss, te_acc = run_epoch(test_loader,  train=False)
    print(f"epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | test loss {te_loss:.4f} acc {te_acc:.4f}")
    print(f"epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | test loss {te_loss:.4f} acc {te_acc:.4f}")
