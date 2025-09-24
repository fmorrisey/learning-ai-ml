import torch
import torch.nn as nn
import torch.nn.functional as F

# Create an instance of MSEloss
mse_loss_fn = nn.MSELoss()

# Example tensors (predicted and target)
predictions = torch.tensor([1.0, 2.5, 3.0])
targets = torch.tensor([1.2, 2.0, 3.5])

# Compute the MSE Loss
loss = mse_loss_fn(predictions, targets)
lossfn = F.mse_loss(predictions, targets)
print(f"MSE Loss using nn.MSELoss: {loss.item()}")
print(f"MSE Loss using F.mse_loss: {loss.item()}")
