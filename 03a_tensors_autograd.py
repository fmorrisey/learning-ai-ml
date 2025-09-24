import torch

# Pick device: GPU if available, otherwise exit. This decides where tensors live.
device = 'cuda' if torch.cuda.is_available() else exit(1) 
deviceName = torch.cuda.get_device_name(0) if torch.cuda.is_available() else exit(1)
print('Device name: ', device, deviceName)

# ----------
# Create a tiny synthetic dataset
# We will make x values and compute y = 2*x + 0.5 plus a little noise.
# The script's job is to re-learn the slope (w) and intercept (b).
# ----------
true_w, true_b = 2.0, 0.5
N = 512
# torch.linspace makes N numbers evenly spaced between -1 and 1
# .unsqueeze(1) turns the 1D list into a column (shape becomes [N,1])
x = torch.linspace(-1, 1, N, device=device).unsqueeze(1)
# Add small random noise to make the problem more realistic
y = true_w * x + true_b + 0.1*torch.randn_like(x)

# ----------
# Parameters we will learn: initialize randomly and ask PyTorch to track them
# requires_grad=True tells PyTorch to remember operations and compute gradients
# ----------
w = torch.randn(1, device=device, requires_grad=True)
b = torch.randn(1, device=device, requires_grad=True)

# Optimizer: tells how to update w and b using gradients. lr is step size.
opt = torch.optim.SGD([w, b], lr=0.5)

# Training loop: repeat predict -> compute loss -> compute gradients -> update
for step in range(201):
    # prediction using current w and b
    pred = w * x + b
    # mean squared error: average of squared differences
    loss = torch.mean((pred - y) ** 2) # MSE

    # zero out gradients from previous step (they accumulate by default)
    opt.zero_grad()
    # compute gradients of loss w.r.t. w and b. This is autograd doing math for you.
    loss.backward()
    # apply the optimizer update to change w and b
    opt.step()

    # Occasionally print progress so you can see learning happen
    if step % 40 == 0:
        # .item() reads a single-number tensor into a Python float
        print(f"step {step:3d}  loss={loss.item():.5f}  w={w.item():.3f}  b={b.item():.3f}")

print("Learned params:", w.item(), b.item())

# Compute and print the final loss explicitly (should not be exactly zero here because we added noise)
final_pred = w * x + b
final_loss = torch.mean((final_pred - y) ** 2)
print("Final MSE loss (after training):", final_loss.item())
