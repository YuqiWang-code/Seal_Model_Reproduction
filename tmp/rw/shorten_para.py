"""Replace the long Related Work paragraph with the concise version."""
import io
import os

DOCS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'docs')
path = os.path.join(DOCS, 'RelatedWork_IRSTD_8方法.md')

PARA = (
    '> Deep learning now dominates infrared small target detection (IRSTD). '
    'Li et al. proposed DNA-Net [1], which densely connects encoder–decoder nodes to preserve tiny targets in deep layers. '
    'Liu et al. presented MSHNet [2], which pairs a plain U-Net with a scale-and-location-sensitive loss. '
    'Xu et al. designed HDNet [3], which enhances multi-scale high-frequency cues to suppress background clutter. '
    'Wu et al. introduced L2SKNet [4], which replaces hand-crafted top-hat filters with learnable saliency kernels. '
    'BGM [5] masks background blocks during training so that the network learns to ignore the background. '
    'Wu et al. further unfolded RPCA into an interpretable network, RPCANet [6], which alternates background estimation and sparse target extraction. '
    'Its extension, RPCANet++ [7], performs the decomposition in the feature domain with a memory-augmented module. '
    'SAM 3 [8] instead formulates promptable concept segmentation from noun phrases or image exemplars. '
    'However, these methods are trained on a single thermal modality with dense masks and are not tailored to few-pixel targets, '
    'motivating box-supervised RGB–thermal detection with promptable foundation models.'
)

text = io.open(path, encoding='utf-8').read()
lines = text.split('\n')
start = next(i for i, ln in enumerate(lines) if ln.startswith('## 二、成段文字'))
end = next(i for i, ln in enumerate(lines) if ln.startswith('## 三、'))
new_block = ['## 二、成段文字', '', PARA, '',
             '约 180 词，8 个方法各一句，每句只保留"解决了什么"，可直接放进 Related Work 的 IRSTD 小节。', '']
lines[start:end] = new_block
io.open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
print('paragraph words:', len(PARA.split()))
