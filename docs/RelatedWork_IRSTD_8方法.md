# Related Work 段落：8 个方法（DNA-Net / MSHNet / RPCANet / HDNet / L2SKNet / BGM / RPCANet++ / SAM 3）

用于论文 Related Work，句式仿 2024–2026 年该方向 CCF-A 期刊/会议的写法：**一句类别开场 → 一个方法一句话 → However 收口引出本文**。每句的机制都已逐条对照官方摘要核对，证据见第四节。

## 一、成段文字（已按摘要校正）

> Deep learning now dominates infrared small target detection (IRSTD). Li et al. proposed DNA-Net [1], which densely connects encoder–decoder nodes to preserve tiny targets in deep layers. Liu et al. presented MSHNet [2], which pairs a plain U-Net with a scale-and-location-sensitive loss. Xu et al. designed HDNet [3], which fuses a dynamic high-pass filter with multi-scale atrous contrast to suppress low-frequency background. Wu et al. introduced L2SKNet [4], which turns the "center-minus-neighbors" saliency prior into learnable local kernels. Liu et al. proposed BGM [5], which masks random background regions during training so that the network learns to ignore the background. Wu et al. further unfolded RPCA into an interpretable network, RPCANet [6], which alternates background estimation and sparse target extraction. Its extension, RPCANet++ [7], adds a memory-augmented module and a deep contrast prior to preserve background features and sharpen object extraction. SAM 3 [8] instead formulates promptable concept segmentation, where a noun phrase or an image exemplar retrieves all instances of a concept. However, these methods are trained on a single thermal modality with dense masks and are not tailored to few-pixel targets, motivating box-supervised RGB–thermal detection built on promptable foundation models.

约 180 词。若不想同段出现两个 "Liu et al."（MSHNet 是 Q. Liu，BGM 是 Y. Liu），可写成 "Q. Liu et al." 与 "Y. Liu et al."。

## 二、核对后的改动（4 处说法被修正）

| 方法 | 改前 | 改后 | 依据 |
|---|---|---|---|
| HDNet | "enhances multi-scale high-frequency cues to suppress background clutter" | "fuses a dynamic high-pass filter with multi-scale atrous contrast to suppress low-frequency background" | 原文里"多尺度"是**空间域**的 MAC（multiscale atrous contrast），"高频/高通"是**频域**的 DHPF；原句把两者混成"多尺度高频"，与摘要不符 |
| L2SKNet | "replaces hand-crafted top-hat filters with learnable saliency kernels" | "turns the 'center-minus-neighbors' saliency prior into learnable local kernels" | 摘要没提 top-hat，它自己的表述是复现 "Center subtracts Neighbors" 显著性原则；改用原文措辞 |
| BGM | "masks background blocks during training" / 未署名 | "masks random background regions during training" / "Liu et al. proposed BGM [5]" | 摘要只说 "background is randomly masked/forgotten"，16×16 分块是代码细节；作者从 IEEE 页查到为 Yongxu Liu 等 |
| RPCANet++ | "performs the decomposition in the feature domain" | "adds a memory-augmented module and a deep contrast prior …" | 论文全文**没有** "feature domain" 说法（只有 "treating the low-rank background as a latent element"），写 feature domain 是我们自己的推断，会被审稿人抓 |
| SAM 3 | "segments all instances" | "retrieves all instances of a concept" | 摘要原文："returns segmentation masks and unique identities for all matching object instances" |

## 三、参考文献（元数据已核）

| 编号 | 方法 | 出处 |
|---|---|---|
| [1] | DNA-Net | B. Li, C. Xiao, L. Wang, Y. Wang, Z. Lin, M. Li, W. An, Y. Guo, "Dense Nested Attention Network for Infrared Small Target Detection," *IEEE TIP*, vol. 32, pp. 1745–1758, 2023（IEEE 9864119，DOI 10.1109/TIP.2022.3199107，在线首发 2022） |
| [2] | MSHNet | Q. Liu, R. Liu, B. Zheng, H. Wang, Y. Fu, "Infrared Small Target Detection with Scale and Location Sensitivity," *CVPR*, 2024, pp. 17490–17499 |
| [3] | HDNet | M. Xu, C. Yu, Z. Li, H. Tang, Y. Hu, L. Nie, "HDNet: A Hybrid Domain Network With Multiscale High-Frequency Information Enhancement for Infrared Small-Target Detection," *IEEE TGRS*, vol. 63, 2025, DOI 10.1109/TGRS.2025.3574962（IEEE 11017756） |
| [4] | L2SKNet | F. Wu, A. Liu, T. Zhang, L. Zhang, J. Luo, Z. Peng, "Saliency at the Helm: Steering Infrared Small Target Detection With Learnable Kernels," *IEEE TGRS*, vol. 63, pp. 1–14, 2025, DOI 10.1109/TGRS.2024.3521947（IEEE 10813615，2024 年末在线） |
| [5] | BGM | Y. Liu, Z. Ma, W. Zhu, N. Li, C. Li, K. Xiong, "Forgetting the Background: A Masking Approach for Enhanced Infrared Small-Target Detection," *IEEE TGRS*, vol. 63, 2025, DOI 10.1109/TGRS.2025.3589602（IEEE 11082367） |
| [6] | RPCANet | F. Wu, T. Zhang, L. Li, Y. Huang, Z. Peng, "RPCANet: Deep Unfolding RPCA Based Infrared Small Target Detection," *WACV*, 2024, pp. 4809–4818 |
| [7] | RPCANet++ | F. Wu, Y. Dai, T. Zhang, Y. Ding, J. Yang, M.-M. Cheng, Z. Peng, "RPCANet++: Deep Interpretable Robust PCA for Sparse Object Segmentation," *IEEE TPAMI*, vol. 48, no. 10, pp. 12420–12440, Oct. 2026, DOI 10.1109/TPAMI.2026.3697815（IEEE 11537388，2026-06-01 上线） |
| [8] | SAM 3 | N. Carion, L. Gustafson, Y.-T. Hu, et al., "SAM 3: Segment Anything with Concepts," *Proc. ICLR*, 2026（OpenReview zF3CrB1iGO，Accept (Poster)；arXiv:2511.16719） |

## 四、逐句证据（官方摘要原文）

| 方法 | 本句说法 | 原文摘录 | 来源 |
|---|---|---|---|
| DNA-Net | 密集连接编解码节点，保住深层小目标 | "we design a dense nested interactive module (DNIM) to achieve progressive interaction among high-level and low-level features. With the repetitive interaction in DNIM, the information of infrared small targets in deep layers can be maintained." | [ar5iv 全文](https://ar5iv.labs.arxiv.org/html/2106.00487)；IEEE 9864119 |
| MSHNet | 纯 U-Net + 尺度位置敏感损失 | "we first propose a novel Scale and Location Sensitive (SLS) loss … Then we design a simple Multi-Scale Head to the plain U-Net (MSHNet). By applying SLS loss to each scale of the predictions our MSHNet outperforms existing state-of-the-art methods by a large margin." | [CVPR 2024 摘要页](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html) |
| HDNet | 频域高通 + 多尺度空洞对比，压制低频背景 | "In the spatial domain, we innovatively introduce a multiscale atrous contrast (MAC) convolution module … In the frequency domain, we have specifically designed the dynamic high-pass filter (DHPF) module, hierarchically calculating low-frequency signal energy and dynamically removing specific low-frequency information … This effectively filters out slowly varying low-frequency backgrounds." | [IEEE 11017756](https://ieeexplore.ieee.org/document/11017756/) |
| L2SKNet | 把"中心减邻域"先验做成可学习局部核 | "This approach implements a learnable local saliency kernel module (LLSKM) that embodies the concept of 'Center subtracts Neighbors,' guiding the network to capture saliency features (points or edges)." | IEEE 10813615 摘要（OpenAlex 取回） |
| BGM | 训练时随机遮背景，逼网络忽略背景 | "The proposed BGM aims to force the network to focus exclusively on the target by masking out irrelevant background information … a proxy training task with masking, enabling the network to simultaneously predict on both the original input and the masked data, where the background is randomly masked/forgotten." | [IEEE 11082367](https://ieeexplore.ieee.org/document/11082367/) |
| RPCANet | 展开 RPCA，交替估计背景 / 提取目标 | "our approach formulates the ISTD task as sparse target extraction, low-rank background estimation, and image reconstruction in a relaxed Robust Principle Component Analysis (RPCA) model. By unfolding the iterative optimization updating steps into a deep-learning framework …" | [WACV 2024 摘要页](https://openaccess.thecvf.com/content/WACV2024/html/Wu_RPCANet_Deep_Unfolding_RPCA_Based_Infrared_Small_Target_Detection_WACV_2024_paper.html) |
| RPCANet++ | MAM 保背景 + DCPM 强化目标提取 | "To mitigate inter-stage transmission loss in BAM, we introduce a Memory-Augmented Module (MAM) to enhance background feature preservation, while a Deep Contrast Prior Module (DCPM) leverages saliency cues to expedite object extraction." | [IEEE 11537388（TPAMI 正式版）](https://ieeexplore.ieee.org/abstract/document/11537388/) |
| SAM 3 | 可提示概念分割：短语/示例 → 该概念全部实例 | "concept prompts, which we define as either short noun phrases … image exemplars, or a combination of both. Promptable Concept Segmentation (PCS) takes such prompts and returns segmentation masks and unique identities for all matching object instances." | [OpenReview（ICLR 2026）](https://openreview.net/forum?id=zF3CrB1iGO)；[arXiv:2511.16719](https://arxiv.org/abs/2511.16719) 第 1 页 |

## 五、投稿前注意

1. **SAM 3 已正式发表**：ICLR 2026（OpenReview 记录 "Decision: Accept (Poster)"，2026-01-26 发布），可放心按会议论文引用 `Proc. ICLR, 2026`，并附 arXiv:2511.16719。
2. **RPCANet++ 已正式发表**：IEEE TPAMI vol. 48, no. 10, pp. 12420–12440, Oct. 2026（DOI 10.1109/TPAMI.2026.3697815），IEEE 版摘要与 arXiv 版一致，MAM/DCPM 说法可直接引。
3. **DNA-Net 年份**：在线首发 2022（DOI 10.1109/TIP.2022.3199107），正式卷期 2023（vol. 32）；本项目表里记 "TIP22"，正文引用建议统一到卷期年。
4. **L2SKNet**：DOI 属 2024 年末在线（10.1109/TGRS.2024.3521947），卷期是 2025（vol. 63），正文写 2025 即可。
