from PIL import Image
import numpy as np
import glob

for p in sorted(glob.glob("/share_datasets/Seal_Model_Reproduction/Seal_data/train/T/*.tif"))[:3]:
    im = Image.open(p)
    a = np.array(im)
    print(p.split("/")[-1], im.mode, a.dtype, a.shape, int(a.min()), int(a.max()))
