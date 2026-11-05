from PIL import Image
src = "Ours/roc_pr/v12_20261105/pd_fa_log.png"
dst = "docs/assets/seal_pd_fa_v12.png"
im = Image.open(src).convert("RGB")
w, h = im.size
target_w = 1800
im2 = im.resize((target_w, round(h * target_w / w)), Image.LANCZOS)
im2.save(dst, optimize=True)
print(src, im.size, "->", im2.size)