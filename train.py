from dataset import make_data_loader

import torch.nn as nn
import torch.optim as optim

import torchvision.models as models

import numpy as np

from torch.backends import cudnn

from bisect import bisect_right
import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


from loss_fn import TripletLoss,softmax_triplet_loss,CrossEntropyLabelSmooth
from eval_fn import accuracy,evaluation
from config import _C as cfg
import os
import progressbar

from model import resnet50
from torch.utils.tensorboard import SummaryWriter
from modeling import build_model

def train(cfg):
    if cfg.MODEL.DEVICE == 'cuda':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')

    train_loader, val_loader, num_query, num_class = make_data_loader(cfg)

    if not os.path.exists(cfg.LOG_DIR):
        os.mkdir(cfg.LOG_DIR)
    writer = SummaryWriter(log_dir=cfg.LOG_DIR)

    # backbone
    model = build_model(cfg,num_class)
    # model = resnet50(num_class,loss='triplet', pretrained=True)
    # model = resnet50(num_class,loss='softmax', pretrained=True,fc_dims=[1000])#,last_stride=2)

    # writer.add_graph(model)

    update_params = model.parameters()
    if cfg.MODEL.FROZEN_FEATURE_EXTRACTION:
        update_params = []
        print('Frozen the feature extraction layers!!!')
        for name,param in model.named_parameters():
            if 'classifier' in name or 'fc' in name:
                # print(name)
                continue
            param.requires_grad = False

        for name,param in model.named_parameters():
            if param.requires_grad == True:
                # print(name)
                update_params.append(param)

    model.to(device)
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        # if device
        clf_ls = CrossEntropyLabelSmooth(num_classes=num_class,
                                         use_gpu=True if cfg.MODEL.DEVICE == 'cuda' else False)
    else:
        clf_ls = nn.CrossEntropyLoss()
    triplet_ls = TripletLoss() #margin=cfg.SOLVER.MARGIN)

    startEp = 0
    pre_state_dict_path = 'output/res50ep65_mLoss1.410033_mAcc100.456374_cetp.pth'
    if os.path.exists(pre_state_dict_path):
        # 方便切换 加载到相应的device上
        checkpoint = torch.load(pre_state_dict_path, map_location=device)
        for name in checkpoint:
            print(name,type(checkpoint[name]))
        model.load_state_dict(checkpoint['model_state_dict'])
        print('loaded weight:',pre_state_dict_path)
    else:
        print(pre_state_dict_path,'not exists')
    print('Evaluation before training ...')
    evaluation(cfg, device, model, val_loader, num_query)

    optimizer = optim.SGD(update_params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    # optimizer = optim.Adam(update_params,lr=cfg.SOLVER.BASE_LR)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    # print(model.conv1.training,model.classifier.training)

    print('start training ......')
    print('sampler:',cfg.DATALOADER.SAMPLER)
    gSteps = 0
    ignore = True
    weight = 1
    for idx_ep in range(startEp,startEp+cfg.SOLVER.MAX_EPOCHS):
        last_loss = 0.0
        last_acc = 0.0
        running_loss = 0.0
        acc = 0
        print('epoch[%d/%d]' % (idx_ep+1,startEp+cfg.SOLVER.MAX_EPOCHS))
        print('ignore triplet-loss:',ignore)
        scheduler.step()
        for i in progressbar.progressbar(range(len(train_loader)),redirect_stdout=True):
            # get the inputs
            gSteps += 1
            data = next(iter(train_loader))
            inputs, labels = data[0],data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            scores,embedings = model(inputs)

            # print(type(scores),type(embedings),type(labels))
            if cfg.DATALOADER.SAMPLER == 'softmax':
                loss = clf_ls(scores, labels)
            elif cfg.DATALOADER.SAMPLER == 'triplet':
                loss,_ap,_an = triplet_ls(scores, labels,normalize_feature=True)
            elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
                triplet_loss,_ap,_an = triplet_ls(embedings,labels)
                clf_loss = clf_ls(scores,labels) # F.cross_entropy(scores,labels)
                # print(type(triplet_loss),type(clf_loss))
                if cfg.SOLVER.CROSS_TRAIN == True and idx_ep % 5 == 0:
                    weight = 1 - ignore
                    ignore = bool(1-ignore) # 取反,每单独训练 几个 clf-loss ，再训练几个混合loss

                loss = weight*triplet_loss + clf_loss
                # 分别记录两种loss
                writer.add_scalars('loss', {'clf_loss': clf_loss.item(),
                                               'triplet_loss': triplet_loss.item()}, gSteps)
            else:
                print('unknown sampler:',cfg.DATALOADER.SAMPLER)

            # 记录loss变化,lr 变化
            writer.add_scalar('loss/all_loss',loss.item(),global_step=gSteps)
            writer.add_scalar('learning-rate',scheduler.get_lr()[0],global_step=gSteps)
            # writer.add_histogram('conv1.weight.data',model.base.conv1.weight.data,global_step=gSteps)
            # writer.add_histogram('classifier.weight.data',model.classifier.weight.data,global_step=gSteps)

            loss.backward()
            optimizer.step()

            # record how grad changes
            # writer.add_histogram('conv1.weight.grad',model.base.conv1.weight.grad,global_step=gSteps) # or layer1.0.conv1.weight
            # writer.add_histogram('classifier.weight.grad',model.classifier.weight.grad,global_step=gSteps)

            # print statistics
            running_loss += loss.item()
            acc += accuracy(scores,labels)[0].item()  # only the top one
            if i % cfg.SOLVER.LOG_PERIOD == cfg.SOLVER.LOG_PERIOD-1:    # print every 2000 mini-batches
                print('last %d batches mean-loss: %.5f mean-acc：%.3f' % (cfg.SOLVER.LOG_PERIOD,
                                         (running_loss-last_loss)/cfg.SOLVER.LOG_PERIOD,
                                         (acc-last_acc)/cfg.SOLVER.LOG_PERIOD))
                last_acc = acc
                last_loss = running_loss
                # print('dist_ap:',_ap)
                # print('dist_an:',_an)
                # print('dist_an - dist_ap',_an - _ap)
                # print('conv1.weight.grad',model.conv1.weight.grad) #or layer1.0.conv1.weight
                # print('classifier.weight.grad',model.classifier.weight.grad)
                # print('loss.grad',loss.grad)
                # print(model.conv1.weight.requires_grad,model.classifier.weight.requires_grad,loss.requires_grad)

        # evaluation after finish EVAL_PERIOD epochs
        if (idx_ep % cfg.SOLVER.EVAL_PERIOD == cfg.SOLVER.EVAL_PERIOD - 1 or idx_ep == cfg.SOLVER.MAX_EPOCHS - 1) :
            all_cmc,mAP = evaluation(cfg,device,model,val_loader,num_query)
            writer.add_scalars('eval',{'rank1':all_cmc[0],
                                       'rank5':all_cmc[4],
                                       'mAP':mAP},idx_ep)
        # save the  model
        if idx_ep % cfg.SOLVER.CHECKPOINT_PERIOD == cfg.SOLVER.CHECKPOINT_PERIOD - 1 :
            filename = 'res50ep%d_mLoss%05f_mAcc%05f_cetp.pth' % (idx_ep+1,running_loss/len(train_loader),acc/len(train_loader))
            if not os.path.exists(cfg.OUTPUT_DIR):
               os.mkdir(cfg.OUTPUT_DIR)
            checkpoint = {'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                          'config': cfg,
                            'loss': loss,
                          'steps': gSteps}
            torch.save(checkpoint,os.path.join(cfg.OUTPUT_DIR,filename))

    print('finish training >.<')


import argparse

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # if cfg.MODEL.DEVICE == "cuda":
    #     os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID  # new add by gu
    # cudnn.benchmark = True
    # print('config device:%s , os.environ:%s' % (cfg.MODEL.DEVICE,os.environ['CUDA_VISIBLE_DEVICES']))
    train(cfg)

if __name__ == '__main__':
    main()
