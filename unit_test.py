
from torch import nn

from solver.lr_scheduler import WarmupMultiStepLR
from solver.build import make_optimizer
from config import _C as cfg


def test_something():
    net = nn.Linear(10, 10)
    optimizer = make_optimizer(cfg, net)
    # lr_scheduler = WarmupMultiStepLR(optimizer, [20, 40], warmup_iters=10)
    lr_scheduler = WarmupMultiStepLR(optimizer, [20,40,60], cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                     cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    for i in range(70):
        lr_scheduler.step()
        for j in range(3):
            print(i, lr_scheduler.get_lr()[0])
            optimizer.step()

test_something()
