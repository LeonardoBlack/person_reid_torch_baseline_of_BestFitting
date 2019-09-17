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



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.

    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res

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

import torch.nn.functional as F

from loss_fn import euclidean_dist,TripletLoss,softmax_triplet_loss,CrossEntropyLabelSmooth

from config import _C as cfg
import os

import time
import progressbar

from model import resnet50

from torch.utils.tensorboard import SummaryWriter


def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea])
        print('using GPU to compute original distance')
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1,-2,feat,feat.t())
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

def evaluation(cfg,device,model,val_loader,num_query):

    since = time.time()
    # features = torch.from_numpy(np.array([]))
    features = []
    pids = []
    camids = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs = data[0]
            pids += data[1]
            camids += data[2]

            inputs = inputs.to(device)
            features.append(model(inputs)[1])

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

    # 交叉计算 query 和 gallery 的距离
    if cfg.TEST.RE_RANKING == 'yes':
        distmat = re_ranking(q_features, g_features, k1=20, k2=6, lambda_value=0.3)
    else:
        distmat = euclidean_dist(q_features, g_features)

    distmat = distmat.cpu().numpy()

    all_cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20)
    print('rank1:', all_cmc[0],
          'mAP:', mAP)
    # writer.flush()

    time_elapsed = time.time() - since
    print('evaluate time elapsed {:.0f}m {:.04f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return all_cmc, mAP

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
    model = resnet50(num_class,loss='triplet', pretrained=True)
    # model = resnet50(num_class,loss='softmax', pretrained=True,fc_dims=[1000])#,last_stride=2)
    # print('^'*10,'model.fc:',model.fc) # always None ,so only one classifier layer is fc

    # for name,param in model.named_parameters():
    #     print(name)
    # return

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

    startEp = 65
    pre_state_dict_path = 'output/cfl1ep_65mLoss_0.499213mAcc37.607230_tp.pth'
    if os.path.exists(pre_state_dict_path):
        # 方便切换 加载到相应的device上
        model.load_state_dict(torch.load(pre_state_dict_path, map_location=device))
    print('Evaluation before training ...')
    evaluation(cfg, device, model, val_loader, num_query)

    optimizer = optim.SGD(update_params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    # optimizer = optim.Adam(update_params,lr=cfg.SOLVER.BASE_LR)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    print(model.conv1.training,model.classifier.training)

    print('start training ......')
    print('sampler:',cfg.DATALOADER.SAMPLER)
    gSteps = 0
    for idx_ep in range(startEp,startEp+cfg.SOLVER.MAX_EPOCHS):
        last_loss = 0.0
        last_acc = 0.0
        running_loss = 0.0
        acc = 0
        print('epoch[%d/%d]' % (idx_ep+1,startEp+cfg.SOLVER.MAX_EPOCHS))
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
                loss = triplet_loss + clf_loss
                # loss = loss.type(torch.cuda.FloatTensor)
                # 分别记录两种loss
                writer.add_scalars('loss', {'clf_loss': clf_loss.item(),
                                               'triplet_loss': triplet_loss.item()}, gSteps)
            else:
                print('unknown sampler:',cfg.DATALOADER.SAMPLER)

            # 记录loss变化,lr 变化
            writer.add_scalar('loss/all_loss',loss.item(),global_step=gSteps)
            writer.add_scalar('learning-rate',scheduler.get_lr()[0],global_step=gSteps)
            writer.add_histogram('conv1.weight.data',model.conv1.weight.data,global_step=gSteps)
            writer.add_histogram('classifier.weight.data',model.classifier.weight.data,global_step=gSteps)

            loss.backward()
            optimizer.step()

            # record how grad changes
            writer.add_histogram('conv1.weight.grad',model.conv1.weight.grad,global_step=gSteps) # or layer1.0.conv1.weight
            writer.add_histogram('classifier.weight.grad',model.classifier.weight.grad,global_step=gSteps)

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
        if (idx_ep % cfg.SOLVER.EVAL_PERIOD == cfg.SOLVER.EVAL_PERIOD - 1 or idx_ep == cfg.SOLVER.MAX_EPOCHS - 1) \
                and running_loss/len(train_loader) < 2:
            all_cmc,mAP = evaluation(cfg,device,model,val_loader,num_query)
            writer.add_scalars('eval',{'rank1':all_cmc[0],
                                       'rank5':all_cmc[4],
                                       'mAP':mAP},idx_ep)
        # save the  model
        if idx_ep % cfg.SOLVER.CHECKPOINT_PERIOD == cfg.SOLVER.CHECKPOINT_PERIOD - 1:
            filename = 'cfl1ep_%dmLoss_%05fmAcc%05f_tp.pth' % (idx_ep+1,running_loss/len(train_loader),acc/len(train_loader))
            if not os.path.exists(cfg.OUTPUT_DIR):
               os.mkdir(cfg.OUTPUT_DIR)
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
