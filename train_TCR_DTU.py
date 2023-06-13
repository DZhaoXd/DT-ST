import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import copy
import datetime
import logging
import time
import math
import numpy as np
from collections import OrderedDict

import torch
import random
import pickle
import torch.nn.functional as F
from PIL import Image
from core.datasets.transform import Compose

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_model, build_feature_extractor, build_classifier 
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU,get_color_pallete
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
import pandas as pd
from tqdm import tqdm
import cv2
from datasets.generate_city_label_info import gen_lb_info
import cv2
from torch import nn


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))

class BinaryCrossEntropy(torch.nn.Module):

    def __init__(self, size_average=True, ignore_index=255):
        super(BinaryCrossEntropy, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_index

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.binary_cross_entropy_with_logits(predict, target.unsqueeze(-1), pos_weight=weight, size_average=self.size_average)
        return loss
        
class WeightEMA(object):
    def __init__(self, params, src_params, alpha):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)

# refer to https://github.com/visinf/da-sac
def pseudo_labels_probs(probs, running_conf, THRESHOLD_BETA, RUN_CONF_UPPER=0.80, ignore_augm=None, discount = True):
    """Consider top % pixel w.r.t. each image"""
    
    RUN_CONF_UPPER = RUN_CONF_UPPER
    RUN_CONF_LOWER = 0.20
    
    B,C,H,W = probs.size()
    max_conf, max_idx = probs.max(1, keepdim=True) # B,1,H,W

    probs_peaks = torch.zeros_like(probs)
    probs_peaks.scatter_(1, max_idx, max_conf) # B,C,H,W
    top_peaks, _ = probs_peaks.view(B,C,-1).max(-1) # B,C
    
    # top_peaks 是一张图上每个类的最大置信度
    top_peaks *= RUN_CONF_UPPER

    if discount:
        # discount threshold for long-tail classes
        top_peaks *= (1. - torch.exp(- running_conf / THRESHOLD_BETA)).view(1, C)

    top_peaks.clamp_(RUN_CONF_LOWER) # in-place
    probs_peaks.gt_(top_peaks.view(B,C,1,1))

    # ignore if lower than the discounted peaks
    ignore = probs_peaks.sum(1, keepdim=True) != 1

    # thresholding the most confident pixels
    pseudo_labels = max_idx.clone()
    pseudo_labels[ignore] = 255

    pseudo_labels = pseudo_labels.squeeze(1)
    #pseudo_labels[ignore_augm] = 255

    return pseudo_labels, max_conf, max_idx

# refer to https://github.com/visinf/da-sac
def update_running_conf(probs, running_conf, THRESHOLD_BETA, tolerance=1e-8):
    """Maintain the moving class prior"""
    STAT_MOMENTUM = 0.9
    
    B,C,H,W = probs.size()
    probs_avg = probs.mean(0).view(C,-1).mean(-1)

    # updating the new records: copy the value
    update_index = probs_avg > tolerance
    new_index = update_index & (running_conf == THRESHOLD_BETA)
    running_conf[new_index] = probs_avg[new_index]

    # use the moving average for the rest (Eq. 2)
    running_conf *= STAT_MOMENTUM
    running_conf += (1 - STAT_MOMENTUM) * probs_avg
    return running_conf

def denormalizeimage(images, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """Denormalize tensor images with mean and standard deviation.
    Args:
        images (tensor): N*C*H*W
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *=255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0,3,1,2))
    return torch.tensor(images)
    
    
def full2weak(feat, target_weak_params, down_ratio=1, nearest=False):
    tmp = []
    for i in range(feat.shape[0]):
        #### rescale
        h, w = target_weak_params['rescale_size'][0][i], target_weak_params['rescale_size'][1][i]
        if nearest:
            feat_ = F.interpolate(feat[i:i+1], size=[int(h/down_ratio), int(w/down_ratio)])
        else:
            feat_ = F.interpolate(feat[i:i+1], size=[int(h/down_ratio), int(w/down_ratio)], mode='bilinear', align_corners=True)
        #### then crop
        y1, y2, x1, x2 = target_weak_params['random_crop_axis'][0][i], target_weak_params['random_crop_axis'][1][i], target_weak_params['random_crop_axis'][2][i], target_weak_params['random_crop_axis'][3][i]
        y1, th, x1, tw = int(y1/down_ratio), int((y2-y1)/down_ratio), int(x1/down_ratio), int((x2-x1)/down_ratio)
        feat_ = feat_[:, :, y1:y1+th, x1:x1+tw]
        if target_weak_params['RandomHorizontalFlip'][i]:
            inv_idx = torch.arange(feat_.size(3)-1,-1,-1).long().to(feat_.device)
            feat_ = feat_.index_select(3,inv_idx)
        tmp.append(feat_)
    feat = torch.cat(tmp, 0)
    return feat

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p, dim=1)
    en = -torch.sum(p * torch.log(p + 1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en
        
def inference(feature_extractor, classifier, image, size, flip=True):
    bs = image.shape[0]
    if flip:
        image = torch.cat([image, torch.flip(image, [3])], 0)
    with torch.no_grad():
        output = classifier(feature_extractor(image)[1])
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[:bs] + output[bs:].flip(2)) / 2
    else:
        output = output
    return output
    
def multi_scale_inference(feature_extractor, classifier, image, tsize, scales=[0.7,1.0,1.3], flip=True):
    feature_extractor.eval()
    classifier.eval()
    output = None
    size = image.shape[-2:]
    for s in scales:
        x = F.interpolate(image, size=(int(size[0]*s), int(size[1]*s)))
        pred = inference(feature_extractor, classifier, x, tsize, flip=False)
        if output is None:
            output = pred
        else:
            output = output + pred
        if flip:
            x_flip = torch.flip(x, [3])
            pred = inference(feature_extractor, classifier, x_flip, tsize, flip=False)
            output = output + pred.flip(3)
    if flip:
        return output/len(scales)/2
    return output/len(scales)
    
        
def train(cfg, local_rank, distributed):
    logger = logging.getLogger("DTST.trainer")
    logger.info("Start training")

    feature_extractor = build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)
    
    classifier = build_classifier(cfg)
    classifier.to(device)


    batch_size = cfg.SOLVER.BATCH_SIZE
    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size())//2
        if not cfg.MODEL.FREEZE_BN:
            feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg2
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()
    
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()
    
    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR*10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    
    output_dir = cfg.OUTPUT_DIR

    local_rank = 0

    start_epoch = 0
    iteration = 0
    
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)
        
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    binary_ce = BinaryCrossEntropy(ignore_index=255)
    
    max_iters = cfg.SOLVER.MAX_ITER
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    feature_extractor.train()
    classifier.train()
    start_training_time = time.time()
    end = time.time()

    classifier_his = copy.deepcopy(classifier).cuda()
    feature_extractor_his = copy.deepcopy(feature_extractor).cuda()
    classifier_his.eval()
    feature_extractor_his.eval()
    
    ###### build dataset 
    top_list = './splits/labeled.txt'
    tail_list = './splits/unlabeled.txt'
    with open(top_list, "r") as handle:
        content = handle.readlines()
        top_list_ids = [i_id.strip().split(' ')[0] for i_id in content]
    with open(tail_list, "r") as handle:
        content = handle.readlines()
        tail_list_ids = [i_id.strip().split(' ')[0] for i_id in content]
    
    ###### Mixup and  rsc init  
    #run_candidate(cfg, feature_extractor_his, classifier_his, local_rank, distributed, init_candidate=True)
    #gen_lb_info(cfg, 'CTR')  ## for mixup 
    #gen_lb_info(cfg, 'CTR_O')  ## for rsc 
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False)
    tgt_train_data.mixer.label_to_file, _ = pickle.load(open(os.path.join(cfg.OUTPUT_DIR, 'CTR.p'), "rb"))

    if distributed:
        src_train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data)
        tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
    else:
        src_train_sampler = None
        tgt_train_sampler = None
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size, shuffle=(tgt_train_sampler is None), num_workers=4, pin_memory=True, sampler=tgt_train_sampler, drop_last=True)

    ###### confident  init 
    #default param in SAC (https://github.com/visinf/da-sac)
    THRESHOLD_BETA = 0.001
    running_conf = torch.zeros(cfg.MODEL.NUM_CLASSES).cuda()
    running_conf.fill_(THRESHOLD_BETA)

    ###### Dynamic teacher init
    if cfg.DTU.DYNAMIC:
        stu_eval_list = []
        stu_score_buffer = []
        res_dict = {'stu_ori':[], 'stu_now':[], 'update_iter':[]}
        
        
    cls_his_optimizer = WeightEMA(
        list(classifier_his.parameters()), 
        list(classifier.parameters()),
        alpha= cfg.DTU.EMA_WEIGHT,
    )  
    feature_extractor_his = copy.deepcopy(feature_extractor).cuda()
    fea_his_optimizer = WeightEMA(
        list(feature_extractor_his.parameters()), 
        list(feature_extractor.parameters()),
        alpha= cfg.DTU.EMA_WEIGHT,
    )      
    
    for rebuild_id in range(255):
        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break
        for i, (tgt_input, y, names, tgt_trans_param, tgt_img_full, mix_label) in enumerate(tgt_train_loader):
            
            data_time = time.time() - end
            
            current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters, power=cfg.SOLVER.LR_POWER)
            for index in range(len(optimizer_fea.param_groups)):
                optimizer_fea.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_cls.param_groups)):
                optimizer_cls.param_groups[index]['lr'] = current_lr*10

            optimizer_fea.zero_grad()
            optimizer_cls.zero_grad()

            tgt_input = tgt_input.cuda(non_blocking=True)
            tgt_size = tgt_input.shape[-2:]
            tgt_img_full = tgt_img_full.cuda(non_blocking=True)
            mix_label = mix_label.cuda()
            tgt_full_size = tgt_img_full.shape[-2:]
            
            ### stu forward
            tgt_pred = classifier(feature_extractor(tgt_input)[1])
            tgt_pred = F.interpolate(tgt_pred, size=tgt_size, mode='bilinear', align_corners=True)

            ######### dy update
            if cfg.DTU.DYNAMIC:
                with torch.no_grad():
                    tgt_pred_full = classifier(feature_extractor(tgt_img_full.clone().detach())[1])
                    output = F.softmax(tgt_pred_full.clone().detach(), dim=1).detach()
                    if cfg.DTU.PROXY_METRIC == 'ENT': 
                        out_max_prob = output.max(1)[0]
                        uc_map_prob = 1- (-torch.mul(out_max_prob, torch.log2(out_max_prob)) * 2)
                        stu_score_buffer.append(uc_map_prob.mean().item())
                        stu_eval_list.append([tgt_img_full.clone().detach().cpu()])
                    elif cfg.DTU.PROXY_METRIC == 'SND':
                        pred1 = output.permute(0, 2, 3, 1)
                        pred1 = pred1.reshape(-1, pred1.size(3))
                        pred1_rand = torch.randperm(pred1.size(0))
                        #select_point = pred1_rand.shape[0]
                        select_point = 100
                        pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                        pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
                        stu_score_buffer.append(pred1_en.item())
                        stu_eval_list.append([tgt_img_full.clone().detach().cpu(), pred1_rand.cpu()])
                    else:
                        print('no support')
                        return
            ###########
                    
             
            #### history model 
            with torch.no_grad():   
                show_images_flag=False
                if show_images_flag:
                    crop_img = full2weak(tgt_img_full, tgt_trans_param)
                    denormalized_image = denormalizeimage(crop_img, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
                    denormalized_image = np.asarray(denormalized_image.cpu().numpy()[2], dtype=np.uint8)
                    denormalized_image = Image.fromarray(denormalized_image.transpose((1,2,0)))
                    denormalized_image.save('tea.png')
                
                    denormalized_image = denormalizeimage(tgt_input, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
                    denormalized_image = np.asarray(denormalized_image.cpu().numpy()[2], dtype=np.uint8)
                    denormalized_image = Image.fromarray(denormalized_image.transpose((1,2,0)))
                    denormalized_image.save('stu.png')
                    
                size = tgt_img_full.shape[-2:]
                tgt_pred_his_full = classifier_his(feature_extractor_his(tgt_img_full)[1], tgt_img_full.shape[-2:])
                
                tgt_prob_his = F.softmax(full2weak(tgt_pred_his_full, tgt_trans_param), dim=1)
                
                # pos label
                running_conf = update_running_conf(F.softmax(tgt_pred_his_full, dim=1), running_conf, THRESHOLD_BETA)
                psd_label, _, _ = pseudo_labels_probs(tgt_prob_his, running_conf, THRESHOLD_BETA)
                
                # neg label
                m_batchsize, C, height, width = tgt_prob_his.size()
                t_neg_label = tgt_prob_his.clone().detach().view(m_batchsize*C, height, width)
                tgt_prob_his = tgt_prob_his.view(m_batchsize*C, height, width)
                thr = 1 / cfg.MODEL.NUM_CLASSES
                t_neg_label[tgt_prob_his > thr] = 255
                t_neg_label[tgt_prob_his <= thr] = 0
                
                # label mix
                psd_label = psd_label * (mix_label==255) + mix_label * ((mix_label!=255))
                uc_map_eln = torch.ones_like(psd_label).float()
                
                if show_images_flag:
                    psd_show = np.asarray(psd_label.cpu().numpy()[2], dtype=np.uint8)
                    psd_show = get_color_pallete(psd_show, "city")
                    psd_show.save('psd_final.png')
            
            st_loss = criterion(tgt_pred, psd_label.long())
            pesudo_p_loss = (st_loss * (-(1-uc_map_eln)).exp() ).mean()
            pesudo_n_loss = binary_ce(tgt_pred.view(m_batchsize*C, 1, height, width), t_neg_label) * 0.5
            
            st_loss = pesudo_p_loss + pesudo_n_loss
            st_loss.backward() 
            
            ### update current model
            optimizer_fea.step()
            optimizer_cls.step()  
        
            ### update history model
            ### eval student perfromance
            if cfg.DTU.DYNAMIC:
                if len(stu_score_buffer) >= cfg.DTU.Query_START and int(len(stu_score_buffer)-cfg.DTU.Query_START) % cfg.DTU.Query_STEP ==0:   
                    all_score = evel_stu(cfg, feature_extractor, classifier, stu_eval_list)
                    compare_res = np.array(all_score) - np.array(stu_score_buffer)
                    if np.mean(compare_res > 0) > 0.5 or len(stu_score_buffer) > cfg.DTU.META_MAX_UPDATE:
                        update_iter = len(stu_score_buffer)

                        cls_his_optimizer.step()
                        fea_his_optimizer.step()
                        
                        res_dict['stu_ori'].append(np.array(stu_score_buffer).mean())
                        res_dict['stu_now'].append(np.array(all_score).mean())
                        res_dict['update_iter'].append(update_iter)
                        
                        df = pd.DataFrame(res_dict)
                        df.to_csv('dyIter_FN.csv')

                        ## reset
                        stu_eval_list = []
                        stu_score_buffer = []

            else:
                if iteration % cfg.DTU.PROXY_METRIC == 0:
                    cls_his_optimizer.step()
                    fea_his_optimizer.step()
           
            ## update
            tgt_train_data.mix_p = (1-running_conf[tgt_train_data.mix_classes]) / ((1-running_conf[tgt_train_data.mix_classes]).sum() )
            
            meters.update(loss_p_loss=pesudo_p_loss.item())
            meters.update(loss_n_loss=pesudo_n_loss.item())
            
            iteration = iteration + 1

            n = tgt_input.size(0)
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iters:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            iter=iteration,
                            meters=str(meters),
                            lr=optimizer_fea.param_groups[0]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )
                    
            if (iteration == cfg.SOLVER.MAX_ITER or iteration % (cfg.SOLVER.CHECKPOINT_PERIOD)==0):
                filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(int(iteration)))
                torch.save({'iteration': iteration, 'feature_extractor': feature_extractor_his.state_dict(), 'classifier':classifier_his.state_dict()}, filename)
                run_test(cfg, feature_extractor_his, classifier_his, local_rank, distributed)

            
            ### re-build candidate and dataloader
            if (iteration+1) % cfg.TCR.UPDATE_FREQUENCY == 0: 
                run_candidate(cfg, feature_extractor_his, classifier_his, local_rank, distributed)
                gen_lb_info(cfg, 'CTR')  ## for mixup 
                gen_lb_info(cfg, 'CTR_O')  ## for rsc 
                tgt_train_data = build_dataset(cfg, mode='train', is_source=False)
                if distributed:
                    tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
                else:
                    tgt_train_sampler = None
                tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size, shuffle=(tgt_train_sampler is None), num_workers=4, pin_memory=True, sampler=tgt_train_sampler, drop_last=True)

                tgt_train_data.mixer.label_to_file, _ = pickle.load(open(os.path.join(cfg.OUTPUT_DIR, 'CTR.p'), "rb"))
                break

            if iteration == cfg.SOLVER.STOP_ITER:
                break
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (cfg.SOLVER.MAX_ITER)
        )
    )

    return feature_extractor, classifier          

def evel_stu(cfg, feature_extractor, classifier, stu_eval_list):
    feature_extractor.eval()
    classifier.eval()
    eval_result = []

    with torch.no_grad():
        for i, (x, permute_index) in enumerate(stu_eval_list):
        
            output = classifier(feature_extractor(x.cuda())[1])
            output = F.softmax(output, dim=1)
            if cfg.DTU.PROXY_METRIC == 'ENT':
                out_max_prob = output.max(1)[0]
                uc_map_prob = 1- (-torch.mul(out_max_prob, torch.log2(out_max_prob)) * 2)
                eval_result.append(uc_map_prob.mean().item())
            elif cfg.DTU.PROXY_METRIC == 'SND':
                pred1 = output.permute(0, 2, 3, 1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                pred1_rand = permute_index
                #select_point = pred1_rand.shape[0]
                select_point = 100
                pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
                eval_result.append(pred1_en.item())
    feature_extractor.train()
    classifier.train()
    return eval_result
    

def run_test(cfg, feature_extractor, classifier, local_rank, distributed):
    logger = logging.getLogger("DTST.tester")
    print("local_rank", local_rank)
    if local_rank==0:
        logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    if distributed:
        feature_extractor, classifier = feature_extractor.module, classifier.module
    torch.cuda.empty_cache()  # TODO check if it helps

    test_data = build_dataset(cfg, mode='test', is_source=False)
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=test_sampler)
    feature_extractor.eval()
    classifier.eval()
    end = time.time()
    with torch.no_grad():
        for i, (x, y, _,) in enumerate(test_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()
            
            size = y.shape[-2:]

            _, fea = feature_extractor(x)
            output = classifier(fea, size)
            output = output.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
            if distributed:
                torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(union), torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if local_rank==0:
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i], accuracy_class[i]))

## CTR:  candidate
## CTR_O:  历史信息，用于更新CTR 和 rsc
def run_candidate(cfg, feature_extractor, classifier, local_rank, distributed, init_candidate=False):
    # 根据 CTR_O 中的历史标签 计算的Rel，
    # 选择TOP ranked样本，mask掉尾部 ranked的样本，并save到 CTR 中； 全部的历史信息则save到 CTR_O
    # 然后 gen_lb_info 生成 CTR.p 文件
    logger = logging.getLogger("tester")
    print("local_rank", local_rank)
    if local_rank==0:
        logger.info('>>>>>>>>>>>>>>>> Run candidate >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    if distributed:
        feature_extractor, classifier = feature_extractor.module, classifier.module
    torch.cuda.empty_cache()  # TODO check if it helps
    dataset_name = cfg.DATASETS.TEST
    ### TTT for output CTR_O
    ### III for output GT
    if init_candidate:
        test_data = build_dataset(cfg, mode='III', is_source=False) 
    else:
        test_data = build_dataset(cfg, mode='TTT', is_source=False)

    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=test_sampler)

    feature_extractor.eval()
    classifier.eval()
    end = time.time()
    

    name_list = []
    if init_candidate:
        predicted_label = np.zeros((len(test_loader), 256, 512))
        predicted_prob = np.zeros((len(test_loader), 256, 512))       
    else:
        predicted_label = np.zeros((len(test_loader), 512, 1024))
        single_iou_list = np.zeros((len(test_loader), cfg.MODEL.NUM_CLASSES))
        
    with torch.no_grad():
        for i, (x, y, name,) in tqdm(enumerate(test_loader)):
            #x = x.cuda(non_blocking=True).half()
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()
            
            size = y.shape[-2:]

            _, fea = feature_extractor(x)
            output = classifier(fea, size)
            probs = F.softmax(output, dim=1)
            pred = output.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(pred, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
            if distributed:
                torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(union), torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            
            if init_candidate:
                prob = probs.max(1)[0]
                predicted_prob[i] = F.interpolate(prob.unsqueeze(0).float(), size=[256, 512]).cpu().numpy().squeeze()
                predicted_label[i] = F.interpolate(pred.unsqueeze(0).float(), size=[256, 512]).cpu().numpy().squeeze()
            else:
                single_iou = intersection / (union + 1e-8)
                single_iou_list[i] = single_iou
                predicted_label[i] = F.interpolate(pred.unsqueeze(0).float(), size=[xx//2 for xx in size]).cpu().numpy().squeeze()
            name_list.append(name)
            
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()

    if init_candidate:
        save_folder = os.path.join(cfg.OUTPUT_DIR, 'CTR')
        mkdir(os.path.dirname(save_folder))
        mkdir(save_folder)
        save_folder_O = os.path.join(cfg.OUTPUT_DIR, 'CTR_O')
        mkdir(save_folder_O)
        thres = []
        for i in range(cfg.MODEL.NUM_CLASSES):
            x = predicted_prob[predicted_label==i]
            if len(x) == 0:
                thres.append(0)
                continue
            x = np.sort(x)
            thres.append(x[int(np.round(len(x)*cfg.TCR.TOPK_CANDIDATE))])
        thres = np.array(thres)
        print('init prob thres is', thres)
        for index in range(len(name_list)):
            name = name_list[index]
            label = predicted_label[index]
            output = np.asarray(label, dtype=np.uint8)
            output = cv2.resize(output, dsize=None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
            mask = get_color_pallete(output, "city")
            mask_filename = name[0] if len(name[0].split("/"))<2 else name[0].split("/")[1]
            mask.save(os.path.join(save_folder_O, mask_filename))
            ### mask
            prob = predicted_prob[index]
            for i in range(cfg.MODEL.NUM_CLASSES):
                label[(prob<thres[i])*(label==i)] = 255
            output = np.asarray(label, dtype=np.uint8)
            output = cv2.resize(output, dsize=None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
            mask = get_color_pallete(output, "city")
            mask_filename = name[0] if len(name[0].split("/"))<2 else name[0].split("/")[1]
            mask.save(os.path.join(save_folder, mask_filename))
            
        print('init_candidate over !!!')
        
    else:
        thres = []
        save_folder = os.path.join(cfg.OUTPUT_DIR, 'CTR')
        save_folder_O = os.path.join(cfg.OUTPUT_DIR, 'CTR_O')
        for i in range(cfg.MODEL.NUM_CLASSES):
            x = single_iou_list[:, i]  
            x = x[x > 0]
            x = np.sort(x)
            if len(x) == 0:
                thres.append(0)
            else:
                thres.append(x[int(np.round(len(x)*cfg.TCR.TOPK_CANDIDATE))])
        thres = np.array(thres)
        print('ReL thres is', thres)            
        for index in range(len(name_list)):
            name = name_list[index]
            label = predicted_label[index]
            mask_filename = name[0] if len(name[0].split("/"))<2 else name[0].split("/")[1]
            t = np.asarray(label, dtype=np.uint8)
            t = cv2.resize(t, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
            mask = get_color_pallete(t, "city")
            mask.save(os.path.join(save_folder_O, mask_filename))
            ReL = single_iou_list[index]
            for i in range(cfg.MODEL.NUM_CLASSES):  ## masking 
                if ReL[i]<thres[i]:
                    label[label==i] = 255  
            output = np.asarray(label, dtype=np.uint8)
            output = cv2.resize(output, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
            mask = get_color_pallete(output, "city")
            mask_filename = name[0] if len(name[0].split("/"))<2 else name[0].split("/")[1]
            mask.save(os.path.join(save_folder, mask_filename))
        print('run_candidate over !!!')
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if local_rank==0:
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        print('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i], accuracy_class[i]))
            print('Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i], accuracy_class[i]))
            


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        RANK = int(os.environ["RANK"])
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            NGPUS_PER_NODE = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        else:
            NGPUS_PER_NODE = torch.cuda.device_count()
        assert NGPUS_PER_NODE > 0, "CUDA is not supported"
        GPU = RANK % NGPUS_PER_NODE
        torch.cuda.set_device(GPU)
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'])
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='tcp://{}:{}'.format(
                                                 master_address, master_port),
                                             rank=RANK, world_size=WORLD_SIZE)
        NUM_GPUS = WORLD_SIZE
        print(f"RANK and WORLD_SIZE in environ: {RANK}/{WORLD_SIZE}")
        # torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(
        #     backend="nccl", init_method="env://"
        # )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("DTST", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    fea, cls = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, fea, cls, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
