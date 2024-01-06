import torch
import torch.nn as nn
import random
import numpy as np

from train import train_epoch
from torch.utils.data import DataLoader
from video_dataset import VideoDataset
from model import generate_model
from opts import parse_opts
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main_test():
    opt = parse_opts()
    print(opt)
    
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")
    
    model = generate_model(opt, device)
    
    transform = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(),
            ToTensorV2()
        ]
    )
    
    dataset = VideoDataset(data_dir=opt.data_dir, num_frames=opt.num_frames, num_levels=opt.num_levels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    print("Dataload finished")
    crnn_params = list(model.parameters())
    optimizer = torch.optim.Adam(crnn_params, lr=opt.lr_rate, weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    print("Start training")
    train_loss, train_acc = train_epoch(model, dataloader, criterion, optimizer, start_epoch, opt.log_interval, device)
    print("train_loss: ", train_loss)
    print("train_acc: ", train_acc)
    
if __name__ == "__main__":
    main_test()