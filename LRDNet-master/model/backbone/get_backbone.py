import torch


def get_backbone(option):
    backbone, channel_list = None, None
    if option['backbone'].lower() == 'swin':
        from model.backbone.swin import SwinTransformer
        backbone = SwinTransformer(img_size=option['trainsize'], embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32], window_size=12)
        pretrained_dict = torch.load(option['pretrain'])["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        backbone.load_state_dict(pretrained_dict)
        channel_list = [128, 256, 512, 1024]
    elif option['backbone'].lower() == 'r50':
        from model.backbone.resnet import ResNet50Backbone
        backbone = ResNet50Backbone()
        channel_list = [256, 512, 1024, 2048]
    elif option['backbone'].lower() == 'replk':
        from model.backbone.replk import create_RepLKNet31B
        backbone = create_RepLKNet31B(drop_path_rate=0.3, num_classes=None, use_checkpoint=False, small_kernel_merged=False)
        pretrained_dict = torch.load(option['pretrain'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        backbone.load_state_dict(pretrained_dict)
        channel_list = [128, 256, 512, 1024]
    elif option['backbone'].lower() == 'res2net':
        from model.backbone.res2net import res2net50_v1b_26w_4s
        backbone = res2net50_v1b_26w_4s()
        pretrained_dict = torch.load(option['pretrain'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        backbone.load_state_dict(pretrained_dict)
        channel_list = [256, 512, 1024, 2048]
    elif option['backbone'].lower() == 'pvtv2':
        from model.backbone.pvtv2 import pvt_v2_b4
        backbone = pvt_v2_b4()
        pretrained_dict = torch.load(option['pretrain'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        backbone.load_state_dict(pretrained_dict)
        channel_list = [64, 128, 320, 512]
    elif option['backbone'].lower() == 'pvtv2b2':
        from model.backbone.pvtv2 import pvt_v2_b2
        backbone = pvt_v2_b2()
        pretrained_dict = torch.load(option['pretrain'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        backbone.load_state_dict(pretrained_dict)
        channel_list = [64, 128, 320, 512]
    elif option['backbone'].lower() == 'p2t':
        from model.backbone.p2t import p2t_large
        backbone = p2t_large()
        pretrained_dict = torch.load(option['pretrain'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        backbone.load_state_dict(pretrained_dict)
        channel_list = [64, 128, 320, 640]
    elif option['backbone'].lower() == 'segformer':
        from model.backbone.segformer import mit_b5
        backbone = mit_b5()
        pretrained_dict = torch.load(option['pretrain'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        backbone.load_state_dict(pretrained_dict)
        channel_list = [64, 128, 320, 512]
    elif option['backbone'].lower() == 'shunted':
        from model.backbone.shunted import shunted_b
        backbone = shunted_b()
        pretrained_dict = torch.load(option['pretrain'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        backbone.load_state_dict(pretrained_dict)
        channel_list = [64, 128, 256, 512]
    elif option['backbone'].lower() == 'twins':
        from model.backbone.twins import alt_gvt_large
        backbone = alt_gvt_large()
        pretrained_dict = torch.load(option['pretrain'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        backbone.load_state_dict(pretrained_dict)
        channel_list = [128, 256, 512, 1024]

    return backbone, channel_list
