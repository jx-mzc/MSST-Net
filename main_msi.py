import torch
import argsParser_msi
from utils import mkExpDir
from model.MSI.mpae import MPAEViT

from trainer_msi import Trainer
from torch import nn
import torch.optim as optim
from dataset import dataloader

args = argsParser_msi.argsParser()
print(args)

def main():
    SEED = 971226
    torch.manual_seed(SEED)
    device = torch.device(args.hsi_device if torch.cuda.is_available() else 'cpu')

    ### make save_dir
    _logger = mkExpDir(args)

    _dataloader = dataloader.get_dataloader(args)

    _model = MPAEViT(args).to(device)

    optimizer = optim.AdamW(_model.parameters(), lr=0.001, betas=(0.9, 0.995), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 300, 1000], gamma=0.1, last_epoch=-1)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, device, optimizer)

    ###  eval / train
    for epoch in range(1, args.num_epochs + 1):
        t.train(current_epoch=epoch)
        if (epoch % args.val_every == 0):
            t.evaluate(current_epoch=epoch)
        scheduler.step()

if __name__ == '__main__':
    main()
