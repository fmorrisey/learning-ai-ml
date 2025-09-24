import torch
import torch.nn as nn
import torch.optim as optim

# Simple linear regression rewritten to use PyTorch and multiple GPUs (DataParallel).

# Detect CUDA devices (your dual P2000s should show up as cuda:0 and cuda:1)
cuda_count = torch.cuda.device_count()
if cuda_count == 0:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')  # main device for model placement
print(f"CUDA devices available: {cuda_count}")
for i in range(cuda_count):
    print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
print("Using device:", device)

# Create the same small dataset but as PyTorch tensors
X = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32).unsqueeze(1)  # shape [6,1]
y = torch.tensor([100, 200, 300, 400, 500, 600], dtype=torch.float32).unsqueeze(1)

# Note: this dataset is an exact linear mapping (y = 100 * x). Because there is no noise,
# a simple linear model can reach a perfect fit where predictions == targets and MSE == 0.
# That is why you may observe the loss drop to exactly zero during training.

# Move data to GPU if available. DataParallel will scatter inputs across GPUs automatically.
X = X.to(device)
y = y.to(device)

# Define a simple linear model y = w*x + b
model = nn.Linear(1, 1, bias=True)

# If you have multiple GPUs, wrap the model in DataParallel so training uses both cards.
if cuda_count > 1:
    # DataParallel will replicate the model on each GPU and split batches along dim 0.
    model = nn.DataParallel(model, device_ids=list(range(cuda_count)))
model.to(device)

# Mean Squared Error and SGD optimizer (similar behavior to the numpy example)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training settings
epochs = 3000 # fewer epochs than the numpy demo but enough to converge on this simple data

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    preds = model(X)            # forward pass
    loss = criterion(preds, y)  # MSE loss

    loss.backward()             # autograd computes gradients
    optimizer.step()            # update parameters

    if epoch % 200 == 0:
        # print progress; .item() converts 1-element tensors to Python numbers
        print(f"epoch {epoch:4d}  loss={loss.item():.6f}")

    # If loss is numerically zero (or extremely small) we can stop early and print diagnostics.
    # This is expected for perfectly linear / noiseless data.
    if loss.item() == 0.0 or loss.item() < 1e-12:
        real_model = model.module if isinstance(model, nn.DataParallel) else model
        print(f"Early stopping at epoch {epoch}: loss={loss.item():.12e}  w={real_model.weight.item():.6f}  b={real_model.bias.item():.6f}")
        break

# Extract learned parameters. If DataParallel was used, the real module is model.module
real_model = model.module if isinstance(model, nn.DataParallel) else model
w_learned = real_model.weight.item()  # weight is shape [1,1], .item() returns float
b_learned = real_model.bias.item()

print("Learned parameters: w =", w_learned, " b =", b_learned)

# Quick sanity: move a test input through the model (on CPU) to see prediction
test_x = torch.tensor([[10.0]])
pred_on_cpu = real_model(test_x.to(device)).cpu().item()
print("Prediction for x=10 ->", pred_on_cpu)
