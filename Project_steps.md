# Your 6-week PyTorch learning path (hands-on, compute-aware)

**Week 0: Python + tooling (1 evening)**

- Set up `uv` or `pip` + venv; install JupyterLab and VS Code server via `code-server` or use SSH + local VS Code remote.
    
- Verify GPU with the test snippet above.
    

**Week 1: Tensors & Autograd**

- Read: PyTorch “60-Minute Blitz”, focus on Tensors, Autograd.
    
- Exercise: Re-implement linear regression from scratch (no nn.Module), then with `nn.Linear`. Compare CPU vs GPU time on small tensors.
    

**Week 2: Vision basics (fits 5 GB VRAM)**

- Dataset: CIFAR-10.
    
- Build a small CNN (≤2–3 M params). Train with **mixed precision** (`torch.cuda.amp`). Try **Albumentations** for augments. Add **TensorBoard**.
    
- Stretch: try **SAM** or **OneCycleLR**; monitor VRAM in `nvidia-smi`.
    

**Week 3: Transfer learning**

- Use a lightweight backbone (e.g., **ResNet18**, **MobileNetV3**, **EfficientNet-B0**) and fine-tune on a small custom set (e.g., your bike/climb photos: helmet/no-helmet classification).
    
- Freeze vs unfreeze layers; measure overfitting and try CutMix/MixUp.
    

**Week 4: Tabular / Time-series (Garmin/Strava-like)**

- Build a simple MLP or **Temporal ConvNet** for forecasting weekly training load or predicting RPE from features. Practice train/val/test splits and metrics.
    

**Week 5: NLP warm-up (VRAM-friendly)**

- Token classification or sentiment with **DistilBERT** (8-bit with **bitsandbytes** if needed). Learn dataset/tokenizer pipelines and gradient accumulation.
    

**Week 6: Distributed basics & profiling**

- Try `torchrun` with **DistributedDataParallel** across both P2000s on a small model. Learn **torch.profiler** to find bottlenecks. Save/restart from checkpoints.
    

**Ongoing craft**

- Logging: TensorBoard or **wandb**.
    
- Repro: seed everything; track versions.
    
- Packaging: move experiments into modules; use Hydra or argparse configs.
    
- Reading: PyTorch recipes + official “Get Started” for the version you install


----

Sources:
[pytorch documentation](https://docs.pytorch.org/docs/stable/index.html)