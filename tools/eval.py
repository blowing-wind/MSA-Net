import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import _init_paths
from config import cfg
from config import update_config
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset.TALDataset import TALDataset
from models.a2net import LocNet
from core.function import train, evaluation
from core.post_process import final_result_process


def parse_args():
    parser = argparse.ArgumentParser(description='SSAD temporal action localization')
    parser.add_argument('--cfg', type=str, help='experiment config file', default='../experiments/A2Net_thumos.yaml')
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(args.cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE

    # data loader
    val_dset = TALDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, drop_last=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.DATASET.PIN_MEMORY)

    model = LocNet(cfg)
    model.cuda()

    #evaluate existing model
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']

    out_df = evaluation(val_loader, model, epoch, cfg)
    final_result_process(out_df, epoch, cfg, flag=0)


if __name__ == '__main__':
    main()


