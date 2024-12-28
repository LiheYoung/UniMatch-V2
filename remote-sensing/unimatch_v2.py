import argparse
from copy import deepcopy
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semicd import SemiCDDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)
    
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW([{'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
                        {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}], 
                        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)

    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)
    criterion_u = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda(local_rank)

    trainset_u = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best_iou, previous_best_acc = 0.0, 0.0
    previous_best_ema_iou, previous_best_ema_acc = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_iou = checkpoint['previous_best_iou']
        previous_best_acc = checkpoint['previous_best_acc']
        previous_best_ema_iou = checkpoint['previous_best_ema_iou']
        previous_best_ema_acc = checkpoint['previous_best_ema_acc']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.7f}, Previous best Changed IoU: {:.2f}, Overall Accuracy: {:.2f} @epoch-{:}, EMA: Changed IoU: {:.2f}, Overall Accuracy: {:.2f} @epoch-{:}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best_iou, previous_best_acc, best_epoch, previous_best_ema_iou, previous_best_ema_acc, best_epoch_ema))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        for i, ((imgA_x, imgB_x, mask_x),
                (imgA_u_w, imgB_u_w, imgA_u_s1, imgB_u_s1,
                 imgA_u_s2, imgB_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):
            
            imgA_x, imgB_x, mask_x = imgA_x.cuda(), imgB_x.cuda(), mask_x.cuda()
            imgA_u_w, imgB_u_w = imgA_u_w.cuda(), imgB_u_w.cuda()
            imgA_u_s1, imgB_u_s1 = imgA_u_s1.cuda(), imgB_u_s1.cuda()
            imgA_u_s2, imgB_u_s2 = imgA_u_s2.cuda(), imgB_u_s2.cuda()
            ignore_mask = ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()

            with torch.no_grad():
                pred_u_w = model(imgA_u_w, imgB_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

            imgA_u_s1[cutmix_box1.unsqueeze(1).expand(imgA_u_s1.shape) == 1] = imgA_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(imgA_u_s1.shape) == 1]
            imgB_u_s1[cutmix_box1.unsqueeze(1).expand(imgB_u_s1.shape) == 1] = imgB_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(imgB_u_s1.shape) == 1]
            imgA_u_s2[cutmix_box2.unsqueeze(1).expand(imgA_u_s2.shape) == 1] = imgA_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(imgA_u_s2.shape) == 1]
            imgB_u_s2[cutmix_box2.unsqueeze(1).expand(imgB_u_s2.shape) == 1] = imgB_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(imgB_u_s2.shape) == 1]

            model.train()
            
            pred_x = model(imgA_x, imgB_x)
            pred_u_s1, pred_u_s2 = model(torch.cat((imgA_u_s1, imgA_u_s2)), torch.cat((imgB_u_s1, imgB_u_s2)), comp_drop=True).chunk(2)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            
            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]
            
            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
            
            loss = (loss_x + loss_u_s) / 2.0
            
            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_u_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, total_mask_ratio.avg))
        
        iou_class, overall_acc = evaluate(model, valloader, cfg, multiplier=14 if cfg['model'] == 'dpt' else None)
        iou_class_ema, overall_acc_ema = evaluate(model_ema, valloader, cfg, multiplier=14 if cfg['model'] == 'dpt' else None)
        
        if rank == 0:
            logger.info('***** Evaluation ***** >>>> Unchanged IoU: {:.2f}, EMA: {:.2f}'.format(iou_class[0], iou_class_ema[0]))
            logger.info('***** Evaluation ***** >>>> Changed IoU: {:.2f}: EMA: {:.2f}'.format(iou_class[1], iou_class_ema[1]))
            logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}, EMA: {:.2f}\n'.format(overall_acc, overall_acc_ema))
            
            writer.add_scalar('eval/unchanged_IoU', iou_class[0], epoch)
            writer.add_scalar('eval/changed_IoU', iou_class[1], epoch)
            writer.add_scalar('eval/overall_accuracy', overall_acc, epoch)
            
            writer.add_scalar('eval/unchanged_IoU_ema', iou_class_ema[0], epoch)
            writer.add_scalar('eval/changed_IoU_ema', iou_class_ema[1], epoch)
            writer.add_scalar('eval/overall_accuracy_ema', overall_acc_ema, epoch)

        is_best = iou_class[1] >= previous_best_iou
        previous_best_iou = max(iou_class[1], previous_best_iou)
        if is_best:
            previous_best_acc = overall_acc
            best_epoch = epoch
        is_best_ema = iou_class_ema[1] >= previous_best_ema_iou
        previous_best_ema_iou = max(iou_class_ema[1], previous_best_ema_iou)
        if is_best_ema:
            previous_best_ema_acc = overall_acc_ema
            best_epoch_ema = epoch
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best_iou': previous_best_iou,
                'previous_best_acc': previous_best_acc,
                'previous_best_ema_iou': previous_best_ema_iou,
                'previous_best_ema_acc': previous_best_ema_acc,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best_ema:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
