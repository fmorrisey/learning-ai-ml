import torch, time
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
n = torch.cuda.device_count()
print("GPU count:", n)
for i in range(n):
    print(f"[{i}] {torch.cuda.get_device_name(i)}")

# tiny GPU matmul to exercise the card
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device='cuda')
    t0 = time.time()
    for _ in range(100):
        y = x @ x
    torch.cuda.synchronize()
    print("Matmul 100× on", x.device, "OK in", round(time.time()-t0, 3), "s")