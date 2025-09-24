import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import autocast, GradScaler
from torchmetrics.classification import MulticlassAccuracy
from pathlib import Path

# Choose device: use GPU if available, otherwise CPU. GPU is faster for big jobs.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

# --------------------
# Data
# --------------------
# Simple pipeline: convert images to tensors and normalize them so values are
# easier for the model to learn from. Normalization centers the data and scales
# it to a consistent range (you don't need to know the math for now).
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Download MNIST dataset (handwritten digits). train=True gives training data.
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# DataLoader turns a dataset into an iterator that gives batches of images.
# batch_size: how many images per step. shuffle=True mixes data each epoch.
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# --------------------
# Model (tiny CNN)
# --------------------
# A tiny Convolutional Neural Network (CNN). Think of it as a small image
# processing machine that learns useful features (edges, shapes) to tell digits apart.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # conv layers look at small patches of the image and learn filters
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 1 input channel (gray), 16 output filters
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # stacked conv to learn deeper features
        # head turns the final feature map into 10 output scores (one per digit)
        self.head  = nn.Sequential(nn.Flatten(), nn.Linear(32*7*7, 128), nn.ReLU(), nn.Linear(128,10))
        # pooling reduces spatial size (makes the model faster and more robust)
        self.pool  = nn.MaxPool2d(2,2)
    def forward(self, x):
        # Forward pass: apply conv -> relu (activation) -> pooling, twice
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # head produces the final 10 scores (called logits)
        return self.head(x)

model = Net().to(device)  # move model to chosen device (GPU/CPU)
if torch.cuda.device_count() > 1:
    # If multiple GPUs are present, this will spread batches across them.
    print("Using DataParallel across", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

# Optimizer: chooses how to change the model's numbers to reduce error.
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
# GradScaler and autocast help with faster mixed-precision training on GPUs.
scaler = GradScaler('cuda' if device == 'cuda' else 'cpu')
# Simple accuracy metric: what fraction of examples were classified correctly.
acc_metric = MulticlassAccuracy(num_classes=10).to(device)


def run_epoch(loader, train=True):
    """Run one pass through the data (train or test).

    - If train=True we update the model; otherwise we only measure performance.
    - Returns average loss and average accuracy for the epoch.
    """
    model.train(train)  # set train/eval mode (affects things like dropout if present)
    total_loss, total_acc, n = 0.0, 0.0, 0
    for xb, yb in loader:
        # Move batch to device. non_blocking=True can speed transfers when using pinned memory.
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        # autocast runs parts of the computation in lower precision when safe,
        # which can be faster on modern GPUs. It's a behind-the-scenes speedup.
        with autocast('cuda', enabled=(device=='cuda')):
            logits = model(xb)                # model predictions (not probabilities)
            loss = F.cross_entropy(logits, yb)  # how wrong the predictions are

        if train:
            # Standard training steps:
            opt.zero_grad(set_to_none=True)      # clear previous gradients
            scaler.scale(loss).backward()        # compute gradients (autograd does this)
            scaler.step(opt)                     # update model parameters
            scaler.update()                      # update scaler for next step

        # Accumulate totals so we can compute averages at the end
        total_loss += loss.item() * xb.size(0)
        total_acc  += acc_metric(logits, yb).item() * xb.size(0)
        n += xb.size(0)
    return total_loss/n, total_acc/n


for epoch in range(1, 4):  # 3 quick epochs
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    te_loss, te_acc = run_epoch(test_loader,  train=False)
    print(f"epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | test loss {te_loss:.4f} acc {te_acc:.4f}")

# Save for later inference demos
save_dir = Path("/models")  # change this to your desired folder
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "mnist_cnn.pt"

# If model is wrapped in DataParallel, save the underlying module's state_dict
state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
torch.save(state_dict, save_path)
print(f"Saved weights to {save_path}")
