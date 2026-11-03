"""Model registry for the DHiF repository.

The upstream file imports every backbone, and the released archive is internally
inconsistent (`model/Unet/utils.py` imports `DualConv_FC_light`, which no longer
exists in `model/basic.py`). Only the MTU backbone used by this experiment is kept
here; the other backbones are unchanged on disk and can be re-enabled by restoring
their import lines.
"""
from model.MTU_Net.MTU_Net import res_UNet as MTUNet
from model.MSHNet.MSHNet import MSHNet
