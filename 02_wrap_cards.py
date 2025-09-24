import torch, torch.nn as nn
net = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.cuda()
