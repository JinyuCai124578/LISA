# read a .pth
import torch
import os
# file_path="/home/bingxing2/ailab/caijinyu/LISA/runs/lisa-em-bio-ft-500step-lr5e-6/meta_log_giou0.276_ciou0.515.pth"
folder_path="/home/bingxing2/ailab/caijinyu/LISA/runs/lisa-em-all-ft-500step-lr1e-6"
for filename in os.listdir(folder_path):
    if filename.endswith(".pth"):
        with open(os.path.join(folder_path, filename), 'rb') as f:
            data = torch.load(f)
            print(filename,data['epoch'])
