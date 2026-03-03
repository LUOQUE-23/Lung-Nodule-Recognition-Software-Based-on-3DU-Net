# 3D U-Net 肺结节分割推理平台

本应用提供一个统一的 Web 界面，用于运行多个已训练的 3D U-Net 模型进行肺结节分割推理。  
平台基于 **LUNA16** 数据集进行训练与测试，支持多种输入格式与三视图可视化结果。

## 功能特点

- 从模型库中选择不同的已训练模型
- 支持上传的输入格式：`.npy`、`.npz`、`.png`、`.jpg`
- 支持加载本地 `.mhd` 文件（需安装 `SimpleITK`）
- 可直接通过索引使用预处理的 LUNA16 数据集样本（`dataset_luna_seg`）
- 运行推理后显示**三视图叠加结果**（横断面 / 冠状面 / 矢状面）
- 自动保存推理输出到本地，包括：
  - `input_volume.npy`　原始输入体积
  - `probability.npy`　　预测概率图
  - `pred_mask.npy`　　　二值化分割掩码
  - `meta.txt`　　　　　　元信息（模型名称、输入尺寸、阈值等）

## 项目整体流程回顾
### 数据准备与训练脚本一览

| 步骤 | 文件名                              | 大小   | 主要功能说明                                                                 |
|------|-------------------------------------|--------|------------------------------------------------------------------------------|
| 1    | `step1_read_annotation.py`          | 2 KB   | 读取原始标注（JSON/XML/CSV/nii等），解析世界坐标、半径、标签等               |
| 2    | `step2_world_to_voxel.py`           | 2 KB   | 世界坐标 → 体素坐标转换（考虑 spacing、origin、direction）                   |
| 3    | `step3_crop_patch.py`               | 2 KB   | 围绕标注点裁剪 3D Patch（通常 32³ 或 64³）                                   |
| 4    | `step4_visualize_patch.py`          | 2 KB   | 可视化裁剪后的 Patch（用于数据检查）                                         |
| 5    | `step5_build_pos_dataset.py`        | 3 KB   | 构建正样本数据集（含结节的 Patch）                                           |
| 6    | `step6_build_neg_dataset.py`        | 4 KB   | 构建负样本数据集（背景 Patch，实现正负平衡）                                 |
| 7    | `step7_train_3dcnn.py`              | 3 KB   | 训练第一阶段 3D CNN 分类器（粗检出正/负 Patch）                              |
| 8    | `step8_generate_mask.py`            | 3 KB   | 用 3D CNN 生成初步分割 mask（伪标签或粗 mask）                               |
| 9    | `step9_build_seg_dataset.py`        | 5 KB   | 构建正式分割数据集（image + mask 对）                                        |
| 10   | `step10_train_3dunet_v2.py`         | 6 KB   | 训练主模型 —— 3D U-Net v2（核心分割训练脚本）                                |
| 11   | `step11_overfit_one_sample.py`      | 3 KB   | 单样本过拟合测试（验证模型与代码正确性）                                     |
| 12   | `step12_infer_and_visualize.py`     | 9 KB   | 单模型推理 + 结果可视化（命令行版本）                                        |
| 13   | `step13_threshold_sweep.py`         | 5 KB   | 阈值扫描，寻找最佳分割阈值（最高 Dice）                                      |
| 14   | `step14_unet_lite_ablation.py`      | 8 KB   | U-Net 轻量化版本消融实验                                                  |
| 15   | `step15_compare_experiments.py`     | 10 KB  | 多实验结果对比（表格 / 图表）                                             |
| 16   | `step16_ablation_experiments.py`    | 16 KB  | 系列消融实验（去模块、改损失等）                                          |
| 17   | `step17_threshold_diagnostics.py`   | 6 KB   | 阈值诊断分析（PR 曲线、F1 变化等）                                           |
| 18   | `step18_threshold_diagnostics_fine.py` | 7 KB | 精细阈值诊断（小步长 + 更多指标）                                            |
| 19   | `step19_sanity_checks.py`           | 4 KB   | 数据与模型的各种健全性检查                                                   |

## 核心数学定义与公式

<img width="631" height="113" alt="image" src="https://github.com/user-attachments/assets/e975eff7-f986-4099-8b56-1c38c3d0c0c0" />
    <img width="590" height="850" alt="image" src="https://github.com/user-attachments/assets/7d369c47-56b4-4fb7-9c77-02e8b87ca0a6" />
    <img width="586" height="489" alt="image" src="https://github.com/user-attachments/assets/8e63068f-1f70-44a6-8e23-5570438b7fe3" />



## 关键算法流程

### 1. 分割数据集构建（正负样本 Patch 采样）
**主要步骤**：

1. 对每条结节标注计算体素中心 → 裁剪 Patch + 生成椭球 mask（正样本）
2. 随机采样背景中心（远离结节）生成负样本
3. 保存为 `images.npy` 与 `masks.npy`
<img width="625" height="443" alt="22c8e2ca712f220bf7f7aa126651f09e" src="https://github.com/user-attachments/assets/8cce2f3d-e294-4824-a0e2-da9670bfa66e" />

### 2. 3D U-Net 训练流程

- 计算正样本权重
- 使用混合损失（BCE + Dice）
- 每个 epoch 在验证集评估 Dice，保存最佳权重
<img width="351" height="361" alt="49cff4169bf2488b35c4d19130742424" src="https://github.com/user-attachments/assets/d2bc5740-acd7-41e1-9a9d-78610f2f1056" />

### 3. 自适应阈值 + 不确定性抑制推理流程

1. 在 Top-K 验证样本上扫描最优阈值 $t^*$
2. 对待推理样本计算概率图 + 体素级不确定性
3. 抑制高不确定区域概率
4. 使用 $t^*$ 二值化得到最终掩码并可视化
<img width="471" height="169" alt="77f08beeca85a502968f0ce577d2f02e" src="https://github.com/user-attachments/assets/e0c373f8-6113-4eff-9fac-ddfda8595365" />

## 本地推理平台使用说明
### 主程序文件

- 应用入口：`app_platform.py`
### 安装依赖

```bash
pip install -r requirements_platform.txt
```
### 运行方式
```Bash
streamlit run app_platform.py
```
### 注意事项
- 输入自动标准化为 [-1, 1] 范围并重采样至 64×64×64
- 若权重文件不存在，界面会显示明确错误提示
- 输出文件自动保存在当前运行目录

