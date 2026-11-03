
from torch.utils.data import DataLoader
import os
import builtins
import argparse
import torch
import torch.distributed as dist
import time
import utils.metrics as Measure
import torch.nn.functional as F
import numpy as np
import FSPNet_model
import dataset
import loss
from model.CamoFormer import CamoFormer
from utils.tools import *
def parse_args():
    parser = argparse.ArgumentParser("FSPNet-Transformer")
    parser.add_argument('--base_lr', default=(1e-4), type=float, help='learning rate')
    parser.add_argument('--batch_size', default=6, type=int, help='batch size per GPU')
    parser.add_argument("--resume", default=None)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--path', type=str, default=r'E:\Mr.Wu\dataset\CodDataset',help='path to train dataset')
    parser.add_argument('--pretrain', type=str, default='E:\Mr.Wu\codes\FSPNet_Weak\checkpoint\Backbone\PVTv2\pvt_v2_b4.pth', help='path to pretrain model')
    parser.add_argument('--ft_for_MoCA', default=None, type=str, help='path to pretrain model')
    
    # DDP configs:
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args

def get_transform(ops=[0,1,2]):
    '''One of flip, translate, crop'''
    op = np.random.choice(ops)
    if op==0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op==1:
        # pp = Translate(0.3)
        pp = Translate(0.15)
    elif op==2:
        pp = Crop(0.7, 0.7)
    return pp

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)
def main(args):
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
            print("args.rank = {}; args.gpu = {}".format(args.rank, args.gpu))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
       
    ### model ###
    net = CamoFormer(cfg=None)
    if args.pretrain:
        encoder = torch.load(args.pretrain)
        net.encoder.load_state_dict(encoder, strict=False)
    # net = FSPNet_model.Model(args.pretrain, img_size=384)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        os.system("nvidia-smi")
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            net = net.cuda(args.gpu)
        else:
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net)
    else:
        net.cuda()
        
    ### optimizer ###
    

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    encoder_param=[]
    decoer_param=[]
    for name, param in net.named_parameters():
        if "encoder" in name:
            encoder_param.append(param)
        else:
            decoer_param.append(param)
    # optimizer = torch.optim.SGD([{"params": encoder_param, "lr":args.base_lr*0.1},{"params":decoer_param, "lr":args.base_lr}], momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam([{"params": encoder_param, "lr":args.base_lr*0.1},{"params":decoer_param, "lr":args.base_lr}])

    ### resume training if necessary ###
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ckpt['state_dict'])
        # optimizer.load_state_dict(ckpt['optimizer'])

    ### Fine tuning for MoCA ###
    if args.ft_for_MoCA is not None:
        ckpt = torch.load(args.ft_for_MoCA, map_location='cpu')
        net.load_state_dict(ckpt)
        print("Fine tuning for MoCA, ckpt from: {}".format(args.ft_for_MoCA))

    
    ### data ###
    Dir = [args.path + r'\train']
    Dataset = dataset.TrainDataset(Dir)
    Datasampler = torch.utils.data.RandomSampler(Dataset, replacement=False)
    Dataloader = DataLoader(Dataset, batch_size=args.batch_size, num_workers=1,
                            collate_fn=dataset.my_collate_fn, sampler=Datasampler, drop_last=True)
    # torch.backends.cudnn.benchmark = True
    TestDir = [args.path + r'\TestDataset\CHAMELEON', args.path + r'\TestDataset\COD10K', args.path + r'\TestDataset\CAMO']
    TestDataset = [dataset.TestDataset(v, size=384) for v in TestDir]
    TestDataloaders = [DataLoader(v, batch_size=1, num_workers=1) for v in TestDataset]


    best_mae = 1
    best_epoch = 0
    ### main loop ###
    for curr_epoch in range(0, 201):

        if curr_epoch==100 or curr_epoch==150:
            for param_group in optimizer.param_groups:
                param_group['lr']= param_group['lr']*0.1
                print("Learning rate:", param_group['lr'])
        # train
        net.train()
        running_loss_all, running_loss_m = 0., 0.
        count = 0
        for data in Dataloader:
            count += 1
            img, label = data['img'].cuda(args.rank), data['label'].cuda(args.rank)
            out1 = net(img)
            all_loss1, m_loss1 = loss.multi_bce(out1, label)
            pre_transform = get_transform([0, 1, 2])
            tr_image = pre_transform(img)
            tr_label = pre_transform(label)
            image_scale = F.interpolate(tr_image, scale_factor=0.5, mode='bilinear', align_corners=True)
            tr_label = F.interpolate(tr_label, scale_factor=0.5, mode='bilinear', align_corners=True)
            out2 = net(image_scale)

            # SaliencyStructure Consistancy
            out_s = pre_transform(out1[4])
            out_s = F.interpolate(out_s, scale_factor=0.5, mode='bilinear', align_corners=True)
            out_s_s = out2[4]
            loss_ssc = SaliencyStructureConsistency(out_s_s, out_s.detach(), 0.85)

            all_loss2, m_loss2 = loss.multi_bce(out2, tr_label)
            all_loss = all_loss2 + all_loss1 + loss_ssc
            m_loss = m_loss2 + m_loss1 + loss_ssc

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            running_loss_all += all_loss.item()
            running_loss_m += m_loss.item()
            if count % 200 == 0 and args.rank == 0:
                print("Epoch:{}, Iter:{}, all_loss:{:.5f}, main_loss:{:.5f}".format(curr_epoch, count, running_loss_all / count, running_loss_m / count))

        # validate
        if (curr_epoch + 1) % 5 == 0:
            maes = []
            for TestDataloader in TestDataloaders:
                net.train(False)
                WFM = Measure.WeightedFmeasure()
                SM = Measure.Smeasure()
                EM = Measure.Emeasure()
                MAE = Measure.MAE()
                with torch.no_grad():
                    st = time.time()
                    for data in TestDataloader:

                        image, mask, name, shape = data['img'].cuda(args.rank), data['label'], data[
                            'name'], \
                            data['shape']
                        _, _, _, _, out = net(image)
                        out = F.interpolate(out, size=shape, mode='bilinear', align_corners=False)
                        pred = out.sigmoid().data.cpu().numpy().squeeze()
                        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # 标准化处理,把数值范围控制到(0,1)
                        mask = mask.numpy().astype(np.float32).squeeze()
                        mask /= (mask.max() + 1e-8)

                        WFM.step(pred=pred * 255, gt=mask * 255)
                        SM.step(pred=pred * 255, gt=mask * 255)
                        EM.step(pred=pred * 255, gt=mask * 255)
                        MAE.step(pred=pred * 255, gt=mask * 255)
                    sm = SM.get_results()['sm'].round(3)
                    adpem = EM.get_results()['em']['adp'].round(3)
                    wfm = WFM.get_results()['wfm'].round(3)
                    mae = MAE.get_results()['mae'].round(3)
                    print('Epoch: {}, MAE: {},  wfm: {}, adpem: {}, sm: {}, dataset: {}, spend_time:{}'.format(
                        curr_epoch, mae, wfm, adpem, sm, TestDataloader.dataset.data_name, (time.time() - st)))
                    maes.append(mae)
                net.train(True)
            if curr_epoch == 1:
                best_mae = maes[0] + maes[1] + maes[2]
            else:
                if maes[0] + maes[1] + maes[2] < best_mae:
                    best_mae = maes[0] + maes[1] + maes[2]
                    best_epoch = curr_epoch

                    torch.save({
                        'state_dict': net.state_dict(),
                        'epoch': curr_epoch
                    }, r'E:\Mr.Wu\codes\FSPNet_Weak\checkpoint\model_checkpoint' + 'Net_epoch_best.pth')
                    print('Save state_dict successfully! Best epoch:{}.'.format(curr_epoch))
            print('the best epoch: {}'.format(best_epoch))
        if args.rank == 0 and curr_epoch % 5 == 0:
            ckpt_save_root = "E:\Mr.Wu\codes\FSPNet_Weak\checkpoint\model_checkpoint"
            if not os.path.exists(ckpt_save_root):
                os.mkdir(ckpt_save_root)
            torch.save(net.state_dict(),
                       ckpt_save_root+"/model_{}_loss_{:.5f}.pth".format(curr_epoch, running_loss_m / count)
                       )


if __name__ == '__main__':
    args = parse_args()
    main(args)