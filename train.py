from dataset import make_data_loader

import torch.nn as nn
import torch.optim as optim

import torchvision.models as models

import numpy as np

from torch.backends import cudnn

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    indices = indices.astype(np.int32)
    # print('*'*10,type(indices),type(g_pids),type(q_pids))
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

import torch
from loss_fn import euclidean_dist,TripletLoss

from config import _C as cfg
import os

import time

def train(cfg):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_query, num_class = make_data_loader(cfg)

    # backbone
    model = models.resnet50(pretrained=True)
    model.to(device)

    # only crossentropy loss
    criterion = nn.CrossEntropyLoss()

    # only triplet-loss
    # criterion = TripletLoss(margin=1.0)[0]

    optimizer = optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)

    print('start training ......')
    for idx_ep in range(cfg.SOLVER.MAX_EPOCHS):
        running_loss = 0.0
        print('epoch[%d/%d]' % (idx_ep+1,cfg.SOLVER.MAX_EPOCHS))
        since = time.time()
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data[0],data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % cfg.SOLVER.LOG_PERIOD == cfg.SOLVER.LOG_PERIOD-1:    # print every 2000 mini-batches
                print('[%4d/%4d] loss: %.5f' % (i + 1,len(train_loader),
                                                             running_loss / cfg.SOLVER.LOG_PERIOD))
                running_loss = 0.0
                time_elapsed = time.time() - since
                print('Training {:.0f}batches elapsed {:.0f}m {:.04f}s'.format(cfg.SOLVER.LOG_PERIOD,
                                                                               time_elapsed // 60, time_elapsed % 60))
                since = time.time()
        # evaluation after finish a epoch,may save the model
        if idx_ep % cfg.SOLVER.EVAL_PERIOD == cfg.SOLVER.EVAL_PERIOD - 1 or idx_ep == cfg.SOLVER.MAX_EPOCHS - 1:

            since = time.time()
            # features = torch.from_numpy(np.array([]))
            features = []
            pids = []
            camids = []

            with torch.no_grad():
                for i,data in enumerate(val_loader):
                    inputs = data[0]
                    pids += data[1]
                    camids += data[2]

                    inputs = inputs.to(device)
                    features.append(model(inputs))

                features = torch.cat(features, dim=0)
                # set copy to gpu
                # pids = pids.to(device)
                # camids = pids.to(device)
                # features.to(device)

            pids = np.asarray(pids)
            camids = np.asarray(camids)

            q_features = features[:num_query]
            q_pids = pids[:num_query]
            q_camids = camids[:num_query]

            g_features = features[num_query:]
            g_pids = pids[num_query:]
            g_camids = camids[num_query:]

            # 交叉计算 query 和 gallery 的欧几里得距离
            distmat = euclidean_dist(q_features,g_features)
            distmat = distmat.cpu().numpy()

            all_cmc,mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20)
            print('rank1:',all_cmc[0],
                  'mAP:',mAP)

            time_elapsed = time.time() - since
            print('evaluate time elapsed {:.0f}m {:.04f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            # save the  model
            if idx_ep % cfg.SOLVER.CHECKPOINT_PERIOD == cfg.SOLVER.CHECKPOINT_PERIOD - 1:
                filename = 'ep_%drank1_%05fmAP%05f' % (idx_ep+1,all_cmc[0],mAP)
                torch.save(model.state_dict(),os.path.join(cfg.OUTPUT_DIR,filename))

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
