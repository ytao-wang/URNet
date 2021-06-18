import os
import torch

import random

from torch.backends import cudnn

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(args.seed)

#torch.cuda.manual_seed(args.seed)
#torch.cuda.manual_seed_all(args.seed)
#random.seed(args.seed)              ##
#cudnn.benchmark = False             ##
#torch.backends.cudnn.deterministic = True

checkpoint = utility.checkpoint(args)


def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()
            if not args.test_only:
                t.time() ##

            checkpoint.done()


if __name__ == '__main__':
    main()
