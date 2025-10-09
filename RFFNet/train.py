import os
import random
import argparse
import numpy as np
import torch
from train_helper_ALTGVT import Trainer

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data-dir', default='/home/zns/xxj/CCTrans/datasets/ucsd_dataset', help='data path')
    parser.add_argument('--dataset', default='ucsd', help='dataset name: mall, ucsd, classroom, bus, canteen')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='the initial learning rate')
    parser.add_argument('--loss', default="ot")
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='the weight decay')
    parser.add_argument('--resume', default='', type=str,
                        help='the path of resume training model')
    parser.add_argument('--dsr', type=int, default=8)
    parser.add_argument('--max-epoch', type=int, default=200,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=3,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=0,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=12,
                        help='the num of training process')
    parser.add_argument('--crop-size', type=int, default=384,
                        help='the crop size of the train image')
    parser.add_argument('--wot', type=float, default=0.1, help='weight on OT loss')
    parser.add_argument('--wtv', type=float, default=0.01, help='weight on TV loss')
    parser.add_argument('--reg', type=float, default=10.0,
                        help='entropy regularization in sinkhorn')
    parser.add_argument('--num-of-iter-in-ot', type=int, default=100,
                        help='sinkhorn iterations')
    parser.add_argument('--norm-cood', type=int, default=0, help='whether to norm cood when computing distance')

    parser.add_argument('--run-name', default='mall', help='run name for wandb interface/logging')
    parser.add_argument('--wandb', default=0, type=int, help='boolean to set wandb logging')
    
    args = parser.parse_args()

    if args.dataset.lower() == 'canteen':
        args.crop_size = 384
    elif args.dataset.lower() == 'classroom':
        args.crop_size = 384
    elif args.dataset.lower() == 'bus':
        args.crop_size = 384
    elif args.dataset.lower() == 'ucsd':
        args.crop_size = 384
    elif args.dataset.lower() == 'mall':
        args.crop_size = 384
    elif args.dataset.lower() == 'fdst':
        args.crop_size = 384
    else:
        raise NotImplementedError
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = False
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()
