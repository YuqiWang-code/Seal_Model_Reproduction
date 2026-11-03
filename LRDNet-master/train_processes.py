from feature_loss import *
from tools import *
import numpy as np


criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').cuda()
loss_lsc = FeatureLoss().cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
l = 0.3

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


def train_loss(image, mask, net, ctx,  mtrsf_prob=1, ops=[0,1,2], w_l2g=0.3, multi_sc=0, l=0.3, sl=1):
    global_step = ctx['global_step']
    sw = ctx['sw']
    second_time = ctx['second_time']

    ######  saliency structure consistency loss  ######
    do_moretrsf = np.random.uniform()<mtrsf_prob
    if do_moretrsf:
        pre_transform = get_transform(ops)
        image_tr = pre_transform(image)
    else:
        image_tr = image
    sc_fct = 0.5
    image_scale = F.interpolate(image_tr, scale_factor=sc_fct, mode='bilinear', align_corners=True)
    out2, _, out3, out4, out5, out6 = net(image, )
    out2_s, _, out3_s, out4_s, out5_s, out6_s = net(image_scale, )


    def out_proc(out2, out3, out4, out5, out6):
        a = [out2, out3, out4, out5, out6]
        a = [i.sigmoid() for i in a]
        a = [torch.cat((1 - i, i), 1) for i in a]
        return a
    out2, out3, out4, out5, out6 = out_proc(out2, out3, out4, out5, out6)
    out2_s, out3_s, out4_s, out5_s, out6_s = out_proc(out2_s, out3_s, out4_s, out5_s, out6_s)

    if not do_moretrsf:
        out2_scale = F.interpolate(out2[:, 1:2], scale_factor=sc_fct, mode='bilinear', align_corners=True)
        out2_s = out2_s[:, 1:2]
    else:
        out2_ss = pre_transform(out2)
        out2_scale = F.interpolate(out2_ss[:, 1:2], scale_factor=0.3, mode='bilinear', align_corners=True)
        out2_s = F.interpolate(out2_s[:, 1:2], scale_factor=0.3/sc_fct, mode='bilinear', align_corners=True)
    loss_ssc = (SaliencyStructureConsistency(out2_s, out2_scale.detach(), 0.85) * (w_l2g + 1) + SaliencyStructureConsistency(out2_s.detach(), out2_scale, 0.85) * (1 - w_l2g)) if sl else 0

    ######   label for partial cross-entropy loss  ######

    gt = mask.squeeze(1).long()
    bg_label = gt.clone()
    fg_label = gt.clone()
    bg_label[gt != 0] = 255  # bg:0, other:255
    fg_label[gt != 1] = 255  # fg:1, other:255

    PCE = criterion(out2, fg_label) + criterion(out2, bg_label)

    ######   local saliency coherence loss (scale to realize large batchsize)  ######
    scale = 0.25
    image_ = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=True)
    sample = {'rgb': image_}
    out2_ = mask_y_hat(out2,mask,scale,second_time)
    loss2_lsc = loss_lsc(out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']

    loss2 = loss_ssc + PCE + l * loss2_lsc

    sw.add_scalar('SSIM', loss_ssc.item() if isinstance(loss_ssc,torch.torch.Tensor) else loss_ssc, global_step=global_step)
    sw.add_scalar('PCE', PCE.item() if isinstance(PCE,torch.torch.Tensor) else PCE, global_step=global_step)
    sw.add_scalar('GLSC', loss2_lsc.item() if isinstance(loss2_lsc,torch.torch.Tensor) else loss2_lsc, global_step = global_step)

    ######  auxiliary losses  ######
    out3_ = mask_y_hat(out3,mask,scale, second_time)
    loss3_lsc = loss_lsc(out3_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    loss3 = criterion(out3, fg_label) + criterion(out3, bg_label) + l * loss3_lsc
    out4_ = mask_y_hat(out4,mask,scale, second_time)
    loss4_lsc = loss_lsc(out4_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    loss4 = criterion(out4, fg_label) + criterion(out4, bg_label) + l * loss4_lsc
    out5_ = mask_y_hat(out5,mask,scale, second_time)
    loss5_lsc = loss_lsc(out5_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    loss5 = criterion(out5, fg_label) + criterion(out5, bg_label) + l * loss5_lsc
    out6_ = mask_y_hat(out6,mask,scale, second_time)
    loss6_lsc = loss_lsc(out6_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
    loss6 = criterion(out6, fg_label) + criterion(out6, bg_label) + l * loss6_lsc

    return loss2, loss3, loss4, loss5, loss6


def mask_y_hat(y_hat, mask, scale, second_time=False):
    y_hat_ = y_hat[:, 1:2]
    y_hat_mask = y_hat_.clone()
    y_hat_mask = F.interpolate(y_hat_mask, size=mask.shape[-1], mode='bilinear', align_corners=True)

    if second_time:
        y_hat_mask_detach = y_hat_mask.detach()
        y_hat_mask[mask == 1] = y_hat_mask_detach[mask == 1]
        y_hat_mask[mask == 0] = y_hat_mask_detach[mask == 0]
    else:
        y_hat_mask[mask == 1] = mask[mask == 1]
        y_hat_mask[mask == 0] = mask[mask == 0]
    y_hat_mask = F.interpolate(y_hat_mask, scale_factor=scale, mode='bilinear', align_corners=True)
    return y_hat_mask