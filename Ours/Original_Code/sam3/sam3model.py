from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import utils
from sam3.adapter import DualLoRALinear, Mode
from sam3.model_builder import build_tracker


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1):
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class CSAGFAdapter(nn.Module):
    """CSAGF-style fusion for SAM3 deep RGB/T features."""

    def __init__(self, in_ch: int = 256, out_ch: int = 256):
        super().__init__()
        self.reduce = ConvBlock(in_ch * 2, in_ch, 3, padding=1)
        self.branch1 = ConvBlock(in_ch, out_ch // 4, 3, padding=1, dilation=1)
        self.branch2 = ConvBlock(in_ch, out_ch // 4, 3, padding=2, dilation=2)
        self.branch3 = ConvBlock(in_ch, out_ch // 4, 3, padding=4, dilation=4)
        self.branch4 = ConvBlock(in_ch, out_ch // 4, 3, padding=6, dilation=6)
        self.pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch // 4, 1, bias=False),
            nn.GroupNorm(1, out_ch // 4),
            nn.ReLU(inplace=True),
        )
        branch_ch = max(1, out_ch // 4)
        self.gfm = CSAGFAttention(branch_ch)
        self.fuse = ConvBlock(branch_ch * 5, out_ch, 1)

    def forward(self, rgb_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.shape[-2:] != t_feat.shape[-2:]:
            t_feat = F.interpolate(t_feat, size=rgb_feat.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(torch.cat([rgb_feat, t_feat], dim=1))
        p1 = self.branch1(x)
        p2 = self.gfm(self.branch2(x))
        p3 = self.gfm(self.branch3(x))
        p4 = self.gfm(self.branch4(x))
        p5 = self.pool_proj(x)
        p5 = F.interpolate(p5, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([p1, p2, p3, p4, p5], dim=1))


class CSAGFAttention(nn.Module):
    """Spatial self-attention used by the CSAGF-style adapter."""

    def __init__(self, in_ch: int):
        super().__init__()
        reduced = max(1, in_ch // 4)
        self.conv_q = ConvBlock(in_ch, reduced, 1)
        self.conv_k = ConvBlock(in_ch, reduced, 1)
        self.conv_v = ConvBlock(in_ch, in_ch, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.conv_q(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.conv_k(x).view(b, -1, h * w)
        attn = self.softmax(torch.bmm(q, k))
        v = self.conv_v(x).view(b, c, h * w)
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return self.gamma * out + x


class Attention(nn.Module):
    def __init__(self, *, dim, fmap_size, heads=4, dim_head=128):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.pos_emb = nn.Parameter(torch.randn(fmap_size[0] * fmap_size[1], inner_dim) * (dim_head ** -0.5))

    def forward(self, fmap):
        b, _, h, w = fmap.shape
        q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
        q = q.view(b, self.heads, -1, h * w).transpose(2, 3)
        k = k.view(b, self.heads, -1, h * w).transpose(2, 3)
        v = v.view(b, self.heads, -1, h * w).transpose(2, 3)
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(2, 3).contiguous().view(b, -1, h, w)
        return out


class BottleBlock(nn.Module):
    def __init__(self, *, dim, fmap_size, dim_out, proj_factor, downsample, heads=4, dim_head=128):
        super().__init__()
        self.downsample = downsample
        self.shortcut = nn.Identity() if dim == dim_out and not downsample else nn.Sequential(
            nn.Conv2d(dim, dim_out, 3 if downsample else 1, stride=2 if downsample else 1,
                      padding=1 if downsample else 0, bias=False),
            nn.BatchNorm2d(dim_out),
        )
        attention_dim = dim_out // proj_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, attention_dim, 1, bias=False),
            nn.BatchNorm2d(attention_dim),
            nn.ReLU(inplace=True),
            Attention(dim=attention_dim, fmap_size=fmap_size, heads=heads, dim_head=dim_head),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(heads * dim_head),
            nn.ReLU(inplace=True),
            nn.Conv2d(heads * dim_head, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
        )
        nn.init.zeros_(self.net[-1].weight)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        shortcut = self.shortcut(x)
        if shortcut.shape[-2:] != out.shape[-2:]:
            shortcut = F.interpolate(shortcut, size=out.shape[-2:], mode="bilinear", align_corners=False)
        return self.activation(out + shortcut)


class BottleStack(nn.Module):
    def __init__(self, *, dim, fmap_size, dim_out=2048, proj_factor=4, num_layers=3, heads=4, dim_head=128):
        super().__init__()
        layers = []
        for i in range(num_layers):
            is_first = i == 0
            layer_fmap_size = fmap_size if is_first else (fmap_size[0] // 2, fmap_size[1] // 2)
            layers.append(BottleBlock(
                dim=dim if is_first else dim_out,
                fmap_size=layer_fmap_size,
                dim_out=dim_out,
                proj_factor=proj_factor,
                downsample=is_first,
                heads=heads,
                dim_head=dim_head,
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    """SSFam-style RGB-T model built on SAM3.

    Flow:
      1) RGB / T pass through the image encoder
      2) Fuse multi-scale features by summation
      3) Decode once without prompts, once with box prompts
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        *,
        image_size: int = 1008,
        t_encoder_type: str = "resnet50",
        resnet_ckpt: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.image_size = int(image_size)
        self.t_encoder_type = t_encoder_type
        self.resnet_ckpt = resnet_ckpt

        tracker = build_tracker(apply_temporal_disambiguation=False, with_backbone=True)
        if ckpt_path:
            utils.load_tracker_checkpoint(tracker, str(ckpt_path))

        self.sam = tracker
        self.prompt_encoder = self.sam.sam_prompt_encoder
        self.mask_decoder_1 = self.sam.sam_mask_decoder
        self.mask_decoder_2 = copy.deepcopy(self.sam.sam_mask_decoder)

        # RGB still uses the original RGB+LoRA path.
        self.dual_qkvs = utils.inject_dual_lora(self.sam.backbone)

        # CSAGF-style deep fusion for the image embedding only.
        # The SAM3 image encoder's deepest feature map is 256 channels in this setup.
        self.deep_fuser = CSAGFAdapter(in_ch=256, out_ch=256)

        # Default: use the Teacher_model-style ResNet50 T encoder.
        # Old T+LoRA can still be enabled via --t_encoder_type lora.
        self._build_t_encoder()

        self.setup()

    def _build_t_encoder(self) -> None:
        if self.t_encoder_type == "lora":
            self.t_encoder_conv1 = None
            return

        # Teacher_model-style thermal encoder: grayscale stem + ResNet backbone + Bottleneck stack.
        resnet_raw_model = models.resnet50(weights=None)
        resnet_ckpt = self.resnet_ckpt or "/share_datasets/Seal_Model_Reproduction/Ours/weights/resnet50-19c8e357.pth"
        if not Path(resnet_ckpt).is_file():
            raise FileNotFoundError(
                f"找不到 ResNet50 权重: {resnet_ckpt}。用 --resnet_ckpt 指定，或下载 "
                "https://download.pytorch.org/models/resnet50-19c8e357.pth"
            )
        resnet_state = torch.load(resnet_ckpt, map_location="cpu", weights_only=False)
        if isinstance(resnet_state, dict) and "state_dict" in resnet_state:
            resnet_state = resnet_state["state_dict"]
        if isinstance(resnet_state, dict):
            resnet_raw_model.load_state_dict(resnet_state, strict=False)
        self.t_encoder_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.t_encoder_conv1.weight.data = torch.unsqueeze(
            torch.mean(resnet_raw_model.conv1.weight.data, dim=1), dim=1
        )
        self.t_encoder_bn1 = resnet_raw_model.bn1
        self.t_encoder_relu = resnet_raw_model.relu
        self.t_encoder_maxpool = resnet_raw_model.maxpool
        self.t_encoder_layer1 = resnet_raw_model.layer1
        self.t_encoder_layer2 = resnet_raw_model.layer2
        self.t_encoder_layer3 = resnet_raw_model.layer3
        # For 1008x1008 inputs, the feature map before layer4 is 63x63 (1/16 scale).
        self.t_encoder_layer4 = BottleStack(
            dim=1024,
            fmap_size=(63, 63),
            dim_out=2048,
            proj_factor=4,
            num_layers=3,
            heads=4,
            dim_head=512,
        )

    def setup(self) -> None:
        for p in self.sam.backbone.parameters():
            p.requires_grad_(False)

        for p in self.prompt_encoder.parameters():
            p.requires_grad_(False)

        for p in self.mask_decoder_1.parameters():
            p.requires_grad_(True)
        for p in self.mask_decoder_2.parameters():
            p.requires_grad_(True)

        # LoRA must stay trainable
        for m in self.modules():
            if isinstance(m, DualLoRALinear):
                for p in m.parameters():
                    p.requires_grad_(True)

        if self.t_encoder_type != "lora":
            for p in [
                self.t_encoder_conv1,
                self.t_encoder_bn1,
                self.t_encoder_layer1,
                self.t_encoder_layer2,
                self.t_encoder_layer3,
                self.t_encoder_layer4,
            ]:
                for param in p.parameters():
                    param.requires_grad_(True)
        else:
            for p in self.sam.backbone.parameters():
                p.requires_grad_(False)

    def _set_lora_mode(self, mode: Mode) -> None:
        for qkv in self.dual_qkvs:
            qkv.set_mode(mode)

    def _encode_rgb(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._set_lora_mode("rgb")
        backbone_out = self.sam.forward_image(x)
        _, vision_feats, _, feat_sizes = self.sam._prepare_backbone_features(backbone_out)
        vision_feats[-1] = vision_feats[-1] + self.sam.no_mem_embed
        feats_2d: list[torch.Tensor] = []
        for feat, (h, w) in zip(vision_feats, feat_sizes):
            b = feat.shape[1]
            feats_2d.append(feat.permute(1, 2, 0).contiguous().view(b, -1, h, w))
        return feats_2d

    def _encode_t_lora(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._set_lora_mode("t")
        backbone_out = self.sam.forward_image(x)
        _, vision_feats, _, feat_sizes = self.sam._prepare_backbone_features(backbone_out)
        vision_feats[-1] = vision_feats[-1] + self.sam.no_mem_embed
        feats_2d: list[torch.Tensor] = []
        for feat, (h, w) in zip(vision_feats, feat_sizes):
            b = feat.shape[1]
            feats_2d.append(feat.permute(1, 2, 0).contiguous().view(b, -1, h, w))
        return feats_2d

    def _encode_t_resnet50(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dim() == 4 and x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        x1 = self.t_encoder_conv1(x)
        x1 = self.t_encoder_bn1(x1)
        x1 = self.t_encoder_relu(x1)
        skip1 = x1

        x1 = self.t_encoder_maxpool(x1)
        x1 = self.t_encoder_layer1(x1)
        x2 = self.t_encoder_layer2(x1)
        skip2 = x2
        x3 = self.t_encoder_layer3(x2)
        x4 = self.t_encoder_layer4(x3)
        return [skip1, skip2, x4]

    @staticmethod
    def _match_feature_channels(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if src.shape[-2:] != ref.shape[-2:]:
            src = F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        if src.shape[1] != ref.shape[1]:
            src = src.mean(dim=1, keepdim=True)
            src = src.expand(-1, ref.shape[1], -1, -1)
        return src

    def _fuse_rgb_t_features(self, rgb_feats: list[torch.Tensor], t_feats: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if not rgb_feats:
            raise RuntimeError("Expected at least one RGB feature map")
        if not t_feats:
            raise RuntimeError("Expected at least one thermal feature map")

        # Keep the original high-resolution fusion path: pairwise sum where thermal
        # features exist, and preserve any extra RGB-only stages as-is.
        high_res_feats: list[torch.Tensor] = []
        for idx, rgb_feat in enumerate(rgb_feats[:-1]):
            if idx < len(t_feats):
                t_feat = self._match_feature_channels(t_feats[idx], rgb_feat)
                high_res_feats.append(rgb_feat + t_feat)
            else:
                high_res_feats.append(rgb_feat)

        # Replace only the deepest embedding with CSAGF-style multi-scale fusion.
        t_deep = self._match_feature_channels(t_feats[-1], rgb_feats[-1])
        image_embed = self.deep_fuser(rgb_feats[-1], t_deep)
        return image_embed, high_res_feats

    @contextmanager
    def _capture_decoder_feature(self, decoder: nn.Module):
        holder: dict[str, Optional[torch.Tensor]] = {"feat": None}

        def _hook(_module, _inputs, output):
            holder["feat"] = output

        # Hook the last activation in output_upscaling (act2), which is always executed
        # in both branches of MaskDecoder.predict_masks.
        target_module = decoder.output_upscaling[-1]
        handle = target_module.register_forward_hook(_hook)
        try:
            yield holder
        finally:
            handle.remove()

    def _decode(
        self,
        image_embed: torch.Tensor,
        high_res_feats: list[torch.Tensor],
        box_prompts: Optional[torch.Tensor] = None,
        mask_prompts: Optional[torch.Tensor] = None,
        decoder: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        decoder = decoder if decoder is not None else self.mask_decoder_1

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_prompts,
            masks=mask_prompts,
        )

        with self._capture_decoder_feature(decoder) as captured:
            out = decoder(
                image_embeddings=image_embed,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats,
            )

        low_res_masks = out[0]
        if low_res_masks.dim() == 4 and low_res_masks.shape[1] > 1:
            low_res_masks = low_res_masks[:, :1]

        decoder_feat = captured["feat"]
        if decoder_feat is None:
            raise RuntimeError("Failed to capture decoder final feature")

        return low_res_masks, decoder_feat

    def encode_images(self, images: torch.Tensor, t_images: torch.Tensor) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        rgb_feats = self._encode_rgb(images.float())

        if self.t_encoder_type == "lora":
            t_feats = self._encode_t_lora(t_images.float())
        elif self.t_encoder_type == "resnet50":
            t_feats = self._encode_t_resnet50(t_images.float())
        else:
            raise ValueError(f"Unsupported t_encoder_type: {self.t_encoder_type}")

        self._set_lora_mode("rgb")

        image_embed, high_res_feats = self._fuse_rgb_t_features(rgb_feats, t_feats)
        return image_embed, high_res_feats

    def decode_noprompt(
        self,
        image_embed: torch.Tensor,
        high_res_feats: list[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._decode(
            image_embed=image_embed,
            high_res_feats=high_res_feats,
            decoder=self.mask_decoder_1,
        )

    def decode_prompt(
        self,
        image_embed: torch.Tensor,
        high_res_feats: list[torch.Tensor],
        box_prompts: Optional[torch.Tensor] = None,
        *,
        mask_prompts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._decode(
            image_embed=image_embed,
            high_res_feats=high_res_feats,
            box_prompts=box_prompts,
            mask_prompts=mask_prompts,
            decoder=self.mask_decoder_2,
        )

    def forward(
        self,
        images: torch.Tensor,
        t_images: torch.Tensor,
        box_prompts: Optional[torch.Tensor] = None,
        mask_prompts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_embed, high_res_feats = self.encode_images(images, t_images)

        mask_1, noprompt_decoder_feat = self.decode_noprompt(
            image_embed,
            high_res_feats,
        )

        mask_2, prompt_decoder_feat = self.decode_prompt(
            image_embed,
            high_res_feats,
            box_prompts=box_prompts,
            mask_prompts=mask_prompts,
        )

        return mask_1, mask_2, noprompt_decoder_feat, prompt_decoder_feat
