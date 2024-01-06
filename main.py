import torch
import torch.nn as nn
import tensorboardX
import os
import random
import numpy as np

from train import train_epoch
from torch.utils.data import DataLoader
from validation import val_epoch
from video_dataset import VideoDataset
from model import generate_model
from opts import parse_opts
from sklearn.model_selection import train_test_split
# Transform
import albumentations as A
from albumentations.pytorch import ToTensorV2

def resume_checkpoint(opt, model, optimizer):
    checkpoint = torch.load(opt.resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model Restored from Epoch {}".format(checkpoint['epoch']))
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch

def main_trainer():
    opt = parse_opts()
    print(opt)
    
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")
    
    # Tensorboard
    summary_writer = tensorboardX.SummaryWriter(log_dir="train_logs")
    
    model = generate_model(opt, device)
    
    transform = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(),
            ToTensorV2()
        ]
    )
    
    full_dataset = VideoDataset(data_dir=opt.data_dir, num_frames=opt.num_frames, num_levels=opt.num_levels, transform=transform)
    train_dataset, val_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    
    # Optimizer
    crnn_params = list(model.parameters())
    optimizer = torch.optim.Adam(crnn_params, lr=opt.lr_rate, weight_decay=opt.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    # Resume from checkpoint
    if opt.resume_path:
        start_epoch = resume_checkpoint(opt, model, optimizer)
    else:
        start_epoch = 0
        
    # Start training
    for epoch in range(start_epoch, opt.n_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, opt.log_interval, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        
        if (epoch) % opt.record_interval == 0:
            # write summary
            summary_writer.add_scalar('loss/train', train_loss, global_step=epoch)
            summary_writer.add_scalar('loss/val', val_loss, global_step=epoch)
            summary_writer.add_scalar('accuracy/train', train_acc, global_step=epoch)
            summary_writer.add_scalar('accuracy/val', val_acc, global_step=epoch)
            
        if (epoch) % opt.checkpoint_interval == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(state, os.path.join('snapshots', f'{opt.model}-Epoch-{epoch}-Loss-{val_loss}.pth'))
            print("Epoch {} model saved!\n".format(epoch))


if __name__ == '__main__':
    main_trainer()