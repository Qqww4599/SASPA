# XNet: Attention combine CNN network for breast ultrasound image segmentation

目前model無正式名稱，屬於開發階段。未來將在此新增相關型介紹。

## Test models

#### ModelLab `<1>`: SwinDeeplabv3Plus: 以deeplabv3+作為骨架，在ASPP module中加入Swin Blocks

![1656665319379](image/README/1656665319379.png)

### ModelLab `<2>`:TransFPN-Unet: ResNet34-Unet架構加入TransFPN

![1656665330995](image/README/1656665330995.png)

## Setting

在此測試中我們使用的配置如下：

##### Training Setting

| images size | Epochs | Loss function            | batchsize | learning rate | weight decay | Accumulation |
| ----------- | ------ | ------------------------ | --------- | ------------- | ------------ | ------------ |
| 128 x 128   | 200    | `Binary cross-entropy` | 8 or 32   | 0.001         | 1e-5         | 4            |

另外為了測試模型對於資料集的學習能力，採用k-fold=5的方式進行訓練。我們將採用兩種資料集分別用於訓練與測試使用:

* Training dataset: 647 breast cancer images
* Testing dataset: 42 breast cancer images

測試模型性能分別以1/5的Training dataset與Testing dataset測試Internal validation與External validation。

## Benchmark

本測試以mIoU、Dice score作為模型效能評斷標準。另外也會用AUC還有Inference time來評斷模型預測結果的信心程度以及模型的推論時間。以下數值皆為External validation。

##### Compare models

| Model               | mIoU           | Dice score     | AUC           | Inference time | Training time | Note |
| ------------------- | -------------- | -------------- | ------------- | -------------- | ------------- | ---- |
| Unet++resnet34      | 85.2%±1.68%   | 90.63%±0.77%  | 96.01%±0.45% | 0.0313         | 10934         |      |
| Segformer           | 79.54%±8.08%  | 86.81%±5.37%  | 95.57%±2.23% | 0.0372         | 15560.04      |      |
| Medical transformer | 47.56%±10.45% | 59.92%±11.34% | 84.6%±3%     | 0.3760         | 53753.18      |      |
| AxialattentionUnet  | 51.93%±16.58% | 64.23%±13.72% | 85.56%±4.86% | 0.0375         | 12720.91      |      |
| FPN_resnet34        | 82.69%±0.32%  | 89.1%±0.21%   | 95.45%±0.04% | 0.0189         | 8745.52       |      |
| Deeplabv3           | 84.51%±0.07%  | 90.37%±0%     | 96.25%±0.1%  | 0.0123         | 10915.14      |      |
| SwinUNETR           | 74.54%±0.02%  | 83.19%±0.08%  | 93.72%±0.05% | 0.0243         | 19736.67      |      |

##### Test models

| mIoU | Dice score | AUC | Inference time | Training time | Note |
| ---- | ---------- | --- | -------------- | ------------- | ---- |
|      |            |     |                |               |      |
|      |            |     |                |               |      |
