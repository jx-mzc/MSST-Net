import torch
import argsParser
from utils import mkExpDir
from model.net import Net
from trainer import Trainer
import torch.optim as optim
from dataset import dataloader

args = argsParser.argsParser()
print(args)

def main():
    SEED = 971226
    torch.manual_seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ### make save_dir
    _logger = mkExpDir(args)

    _dataloader = dataloader.get_dataloader(args)

    _model = Net(args).to(device)
    #print(_model)

    hsi_state_1 = _model.spectral_layers_1.state_dict()
    hsi_state_2 = _model.spectral_layers_2.state_dict()
    hsi_state_3 = _model.spectral_layers_3.state_dict()
    hsi_ckpt_state_dict = {}
    state_dict_1 = torch.load(args.hsi_model_path_1, map_location=args.device)
    state_dict_2 = torch.load(args.hsi_model_path_2, map_location=args.device)
    state_dict_3 = torch.load(args.hsi_model_path_3, map_location=args.device)

    for key, value in state_dict_1.items():
        if 'encoder.' in key:
            hsi_ckpt_state_dict[key[8:]] = value
    hsi_state_1.update(hsi_ckpt_state_dict)
    _model.spectral_layers_1.load_state_dict(hsi_state_1)

    for key, value in state_dict_2.items():
        if 'encoder.' in key:
            hsi_ckpt_state_dict[key[8:]] = value
    hsi_state_2.update(hsi_ckpt_state_dict)
    _model.spectral_layers_2.load_state_dict(hsi_state_2)

    for key, value in state_dict_3.items():
        if 'encoder.' in key:
            hsi_ckpt_state_dict[key[8:]] = value
    hsi_state_3.update(hsi_ckpt_state_dict)
    _model.spectral_layers_3.load_state_dict(hsi_state_3)


    msi_state_16 = _model.spatial_layers_16.state_dict()
    msi_state_8 = _model.spatial_layers_8.state_dict()
    msi_state_32 = _model.spatial_layers_32.state_dict()
    msi_ckpt_state_dict = {}
    state_dict_16 = torch.load(args.msi_model_path_16, map_location=args.device)
    state_dict_8 = torch.load(args.msi_model_path_16, map_location=args.device)
    state_dict_32 = torch.load(args.msi_model_path_16, map_location=args.device)

    for key, value in state_dict_16.items():
        if 'encoder.' in key:
            msi_ckpt_state_dict[key[8:]] = value
    msi_state_16.update(msi_ckpt_state_dict)
    _model.spatial_layers_16.load_state_dict(msi_state_16)

    for key, value in state_dict_8.items():
        if 'encoder.' in key:
            msi_ckpt_state_dict[key[8:]] = value
    msi_state_8.update(msi_ckpt_state_dict)
    _model.spatial_layers_8.load_state_dict(msi_state_8)

    for key, value in state_dict_32.items():
        if 'encoder.' in key:
            msi_ckpt_state_dict[key[8:]] = value
    msi_state_32.update(msi_ckpt_state_dict)
    _model.spatial_layers_32.load_state_dict(msi_state_32)


    optimizer = optim.AdamW(_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.step, gamma=args.gamma, last_epoch=-1)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, device, optimizer)

    ###  train

    for epoch in range(1, args.num_epochs + 1):
        t.train(current_epoch=epoch)
        if (epoch % args.val_every == 0):
            t.evaluate(current_epoch=epoch)
        scheduler.step()

if __name__ == '__main__':
    main()
