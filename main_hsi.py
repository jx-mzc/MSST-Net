import torch
import argsParser_hsi
from utils import mkExpDir
from model.HSI.mbae import MBAEViT
from trainer_hsi import Trainer
import torch.optim as optim
from dataset import dataloader

args = argsParser_hsi.argsParser()
print(args)

def main():
    SEED = 971226
    torch.manual_seed(SEED)
    device = torch.device(args.hsi_device if torch.cuda.is_available() else 'cpu')

    ### make save_dir
    _logger = mkExpDir(args)

    _dataloader = dataloader.get_dataloader(args)

    _model = MBAEViT(args).to(device)

    optimizer = optim.AdamW(_model.parameters(), lr=args.lr, betas=(0.9, 0.995), weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.step, gamma=args.gamma, last_epoch=-1)

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
