from torch.utils.data import DataLoader
from dataset.datasets import CaveDataset, HarvardDataset, WDCMDataset, YREDataset


def get_dataloader(args):
    if (args.dataset == 'cave'):
        data_train = CaveDataset(args.dataset_dir, type='train')
        trainLoader = DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        data_eval = CaveDataset(args.dataset_dir, type='eval')
        evalLoader = DataLoader(data_eval, batch_size=1, num_workers=args.num_workers)
        dataloader = {'train': trainLoader, 'eval': evalLoader}
    elif (args.dataset == 'harvard'):
        data_train = HarvardDataset(args.dataset_dir, type='train')
        trainLoader = DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        data_eval = HarvardDataset(args.dataset_dir, type='eval')
        evalLoader = DataLoader(data_eval, batch_size=1, num_workers=args.num_workers)
        dataloader = {'train': trainLoader, 'eval': evalLoader}
    elif (args.dataset == 'WDCM'):
        data_train = WDCMDataset(args.dataset_dir, type='train')
        trainLoader = DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        data_eval = WDCMDataset(args.dataset_dir, type='eval')
        evalLoader = DataLoader(data_eval, batch_size=1, num_workers=args.num_workers)
        dataloader = {'train': trainLoader, 'eval': evalLoader}
    elif (args.dataset == 'YRE'):
        data_train = YREDataset(args.dataset_dir, type='train')
        trainLoader = DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        data_eval = YREDataset(args.dataset_dir, type='eval')
        evalLoader = DataLoader(data_eval, batch_size=1, num_workers=args.num_workers)
        dataloader = {'train': trainLoader, 'eval': evalLoader}
    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader