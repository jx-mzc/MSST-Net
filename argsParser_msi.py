import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def argsParser():
    parser = argparse.ArgumentParser()


    ### log setting

    parser.add_argument('--save_dir', type=str, default='./train_msi/cave/8/16',
                        help='Directory to save log, arguments, models and images')
    parser.add_argument('--reset', type=str2bool, default=True,
                        help='Delete save_dir to create a new one')
    parser.add_argument('--log_file_name', type=str, default='train.log',
                        help='Log file name')
    parser.add_argument('--logger_name', type=str, default='train',
                        help='Logger name')

    ### device setting
    parser.add_argument('--cpu', type=str2bool, default=False,
                        help='Use CPU to run code')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='The number of GPU used in training')

    ### dataset setting
    parser.add_argument('--dataset', type=str, default='cave', choices=['cave', 'harvard', 'WDCM', 'YRE'],
                        help='Which dataset to train and test')
    parser.add_argument('--dataset_dir', type=str, default='',
                        help='Directory of dataset')
    parser.add_argument('--ratio', type=int, default=8)

    ### dataloader setting
    parser.add_argument('--num_workers', type=int, default=8,
                        help='The number of workers when loading data')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='Learning rate weight decay')
    parser.add_argument('--step', type=list, default=[100, 300, 1000],
                        help='step size for step decay')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Learning rate decay factor for step decay')

    ### model setting
    parser.add_argument('--msi_channel', type=int, default=3)
    parser.add_argument('--msi_embed_dim', type=int, default=256)
    parser.add_argument('--msi_mask_ratio', type=int, default=0.5)
    parser.add_argument('--msi_heads', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--msi_num_layers', type=int, default=4)

    ### pre-training setting
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=5000,
                        help='The number of training epochs')
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save period')
    parser.add_argument('--val_every', type=int, default=10,
                        help='Validation period')
    parser.add_argument('--test', type=str2bool, default=False,
                        help='Test mode')


    args = parser.parse_args()

    return args
