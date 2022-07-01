# XNet: Attention combine CNN network for breast ultrasound image segmentation

目前model無正式名稱，屬於開發階段。未來將在此新增相關型介紹。

## Test models

#### ModelLab `<1>`: SwinDeeplabv3Plus: 以deeplabv3+作為骨架，在ASPP module中加入Swin Blocks

![1656662257154.png](image/README/1656662257154.png)


### ModelLab `<2>`:TransFPN-Unet: ResNet34-Unet架構加入TransFPN

![1656662759542.png](image/README/1656662759542.png)

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

| Model               | mIoU | Dice score | AUC | Inference time | Training time | Note |
| ------------------- | ---- | ---------- | --- | -------------- | ------------- | ---- |
| Unet++resnet34      |      |            |     |                |               |      |
| Segformer           |      |            |     |                |               |      |
| Medical transformer |      |            |     |                |               |      |
| AxialattentionUnet  |      |            |     |                |               |      |
| FPN_resnet34        |      |            |     |                |               |      |
| Deeplabv3           |      |            |     |                |               |      |
| SwinUNETR           |      |            |     |                |               |      |
