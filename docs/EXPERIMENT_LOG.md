# 实验记录（3DUNET）

## 概要
- 目标：提升分割结果（降低假阳性，提升 Dice）
- 当前结论：可视化显示稳步提升（Pred 覆盖更贴近 GT）

## 数据集
- 数据来源：LUNA16 + annotations.csv
- Patch 大小：64^3
- 采样策略：正样本 + 负样本（远离结节中心）
- 正负数量：NUM_POS=300，NUM_NEG=300

## 模型与训练
### 基线（旧）
- 模型：UNet3DLite（8/16）
- 训练：BCE + Dice，阈值多为 0.1

### 改进版（当前）
- 模型：UNet3D（16/32/64）
- 训练脚本：step10_train_3dunet_v2.py
- 训练设置：EPOCHS=40，LR=1e-3，val split=0.15
- 保存权重：
  - best_model_unet3d.pth（val Dice 最优）
  - unet3d_v3.pth（最后一轮）

### 微调（当前）
- 脚本：step14_unet_lite_ablation.py
- 预训练权重：best_model_unet3d.pth
- 冻结策略：enc1 冻结，enc2 + bottleneck 解冻
- 阈值策略：Top20 + 自动阈值，输出 soft_dice 与 mean_prob

## 评估与可视化
- 可视化脚本：step12_infer_and_visualize.py
- 权重：best_model.pth（最新 best）
- 结果：Pred overlay 形状与 GT 更接近，误检明显减少
- 代表图：seg_outputs/ 目录下的三视图 PNG

## 关键问题与处理
- 结构不匹配：best_model.pth 来自 Lite 模型，UNet3D 加载会报错
  - 解决：UNet3D 统一使用 best_model_unet3d.pth / unet3d_v3.pth
- Dice 停滞：阈值固定导致全 0 预测
  - 解决：加入自动阈值 + soft_dice + mean_prob 诊断

## 后续建议（可选）
1) 固定使用 best_model_unet3d.pth 做可视化对比
2) 若 Dice 仍偏低：加入更强数据增强（随机旋转/伪采样）
3) 建立固定验证集并记录指标趋势


## Dice Comparison Experiments
### 2025-12-29 15:32
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\compare_dice_topk.png

| Model | Weights | BestThr | TopKMeanDice |
| --- | --- | --- | --- |
| UNet3D | best_model_unet3d.pth | 0.10 | 0.0766 |
| UNet3DLite | unet3d_lite_v2.pth | 0.50 | 0.2575 |

## Dice Comparison Experiments
### 2025-12-29 15:35
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice/IoU/Precision/Recall/F1
- Threshold selection: best Dice among thresholds
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\compare_metrics_topk.png

| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UNet3D | best_model_unet3d.pth | 0.10 | 0.0766 | 0.0400 | 0.0400 | 1.0000 | 0.0766 |
| UNet3DLite | unet3d_lite_v2.pth | 0.50 | 0.2575 | 0.1526 | 0.1586 | 0.8260 | 0.3075 |

## Dice Comparison Experiments
### 2025-12-29 15:39
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice/IoU/Precision/Recall/F1
- Threshold selection: best Dice among thresholds
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\compare_metrics_topk.png

| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UNet3D | best_model_unet3d.pth | 0.10 | 0.0766 | 0.0400 | 0.0400 | 1.0000 | 0.0766 |
| UNet3DLite | unet3d_lite_v2.pth | 0.50 | 0.2575 | 0.1526 | 0.1586 | 0.8260 | 0.3074 |

## Ablation Experiments Workflow
- Script: step16_ablation_experiments.py
- Outputs: seg_outputs_thresh/ablation_*.png
- Log: auto-append to this file (English)

## Ablation: Threshold Strategy
### 2025-12-29 15:44
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice/IoU/Precision/Recall/F1
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\ablation_threshold.png
- Notes: Baseline uses best Dice threshold; ablation uses fixed 0.30.

| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | best_model_unet3d.pth | 0.10 | 0.0766 | 0.0400 | 0.0400 | 1.0000 | 0.0766 |
| FixedThr0.30 | best_model_unet3d.pth | 0.30 | 0.0766 | 0.0400 | 0.0400 | 1.0000 | 0.0766 |

## Ablation: Channel Width
### 2025-12-29 15:46
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice/IoU/Precision/Recall/F1
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\ablation_channels.png
- Notes: Baseline uses base_ch=16; ablation uses base_ch=8.

| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | best_model_unet3d.pth | 0.10 | 0.0766 | 0.0400 | 0.0400 | 1.0000 | 0.0766 |
| UNet3D-Ch8 | ablation_unet3d_ch8.pth | 0.50 | 0.4108 | 0.2649 | 0.2771 | 0.8968 | 0.4108 |

## Ablation: Loss Function
### 2025-12-29 15:50
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice/IoU/Precision/Recall/F1
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\ablation_loss.png
- Notes: Baseline uses BCE+Dice; ablation uses BCE-only.

| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | best_model_unet3d.pth | 0.10 | 0.0766 | 0.0400 | 0.0400 | 1.0000 | 0.0766 |
| BCE-only | ablation_unet3d_bce_only.pth | 0.50 | 0.4275 | 0.2834 | 0.2946 | 0.8664 | 0.4775 |

## Dice Comparison Experiments
### 2025-12-29 16:07
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice/IoU/Precision/Recall/F1
- Threshold selection: best Dice among thresholds
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\compare_metrics_topk.png

| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UNet3D | best_model_unet3d.pth | 0.10 | 0.0766 | 0.0400 | 0.0400 | 1.0000 | 0.0766 |
| UNet3DLite | unet3d_lite_v2.pth | 0.50 | 0.2575 | 0.1526 | 0.1586 | 0.8260 | 0.3074 |

## Dice Comparison Experiments
### 2025-12-29 16:15
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice/IoU/Precision/Recall/F1
- Threshold selection: best Dice among thresholds
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\compare_metrics_topk.png

| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UNet3D | best_model_unet3d.pth | 0.10 | 0.0766 | 0.0400 | 0.0400 | 1.0000 | 0.0766 |
| UNet3DLite | unet3d_lite_v2.pth | 0.50 | 0.2575 | 0.1526 | 0.1586 | 0.8260 | 0.3075 |

## Ablation: Threshold Strategy
### 2025-12-29 16:15
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice/IoU/Precision/Recall/F1
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\ablation_threshold.png
- Notes: Baseline uses best Dice threshold; ablation uses fixed 0.30.

| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | unet3d_lite_v2.pth | 0.50 | 0.2575 | 0.1526 | 0.1586 | 0.8260 | 0.3075 |
| FixedThr0.30 | unet3d_lite_v2.pth | 0.30 | 0.2028 | 0.1159 | 0.1174 | 0.8940 | 0.2029 |

## Ablation: Channel Width
### 2025-12-29 16:17
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice/IoU/Precision/Recall/F1
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\ablation_channels.png
- Notes: Baseline uses base_ch=8; ablation uses base_ch=4.

| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | unet3d_lite_v2.pth | 0.50 | 0.2575 | 0.1526 | 0.1586 | 0.8260 | 0.3075 |
| UNet3DLite-Ch4 | ablation_unet3d_lite_ch4.pth | 0.50 | 0.3349 | 0.2062 | 0.2158 | 0.8648 | 0.3349 |

## Ablation: Loss Function
### 2025-12-29 16:18
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice/IoU/Precision/Recall/F1
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\ablation_loss.png
- Notes: Baseline uses BCE+Dice; ablation uses BCE-only.

| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | unet3d_lite_v2.pth | 0.50 | 0.2575 | 0.1526 | 0.1586 | 0.8260 | 0.3075 |
| BCE-only | ablation_unet3d_lite_bce_only.pth | 0.50 | 0.3006 | 0.1822 | 0.1870 | 0.9148 | 0.3006 |

## Ablation: Threshold Strategy
### 2025-12-29 16:40
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice/IoU/Precision/Recall/F1
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\ablation_threshold.png
- Notes: Baseline uses best Dice threshold; ablation uses fixed 0.30.

| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | ablation_unet3d_lite_ch4.pth | 0.50 | 0.3349 | 0.2062 | 0.2158 | 0.8648 | 0.3349 |
| FixedThr0.30 | ablation_unet3d_lite_ch4.pth | 0.30 | 0.2894 | 0.1730 | 0.1763 | 0.9318 | 0.2894 |

## Ablation: Channel Width
### 2025-12-29 16:42
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice/IoU/Precision/Recall/F1
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\ablation_channels.png
- Notes: Baseline uses base_ch=4; ablation uses base_ch=2.

| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | ablation_unet3d_lite_ch4.pth | 0.50 | 0.3349 | 0.2062 | 0.2158 | 0.8648 | 0.3349 |
| UNet3DLite-Ch2 | ablation_unet3d_lite_ch2.pth | 0.50 | 0.2947 | 0.1789 | 0.2391 | 0.7963 | 0.2947 |

## Ablation: Loss Function
### 2025-12-29 16:43
- Dataset: dataset_luna_seg (Top20 fg)
- Thresholds: [0.1, 0.2, 0.3, 0.4, 0.5]
- Metric: TopK mean Dice/IoU/Precision/Recall/F1
- Figure: D:\desktop\3DUNET\seg_outputs_thresh\ablation_loss.png
- Notes: Baseline uses BCE+Dice (Ch4); ablation uses BCE-only (Ch4).

| Model | Weights | BestThr | Dice | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | ablation_unet3d_lite_ch4.pth | 0.50 | 0.3349 | 0.2062 | 0.2158 | 0.8648 | 0.3349 |
| BCE-only | ablation_unet3d_lite_ch4_bce_only.pth | 0.50 | 0.3154 | 0.1917 | 0.2049 | 0.7957 | 0.3154 |
