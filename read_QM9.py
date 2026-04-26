import torch
data=torch.load("qm9_3d_fgn_dataset.pt",weights_only=False)
print(data[0].x)