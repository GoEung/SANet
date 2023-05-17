# AI 특론 중간고사

## Introduction
MICCAI 2021에 accept된 "Shallow Attention Network for Polyp Segmentation"(<https://arxiv.org/abs/2108.00882>)의 구현 코드입니다.

![image](https://github.com/GoEung/SANet/assets/53440224/57baaa53-5878-4513-94a3-bc65fe8f2e48)

SAM이라는 Shallow Attention Module을 사용하여 object의 foreground에 집중할 수 있는 방법을 제시한 논문입니다.

```
    out2, out3, out4, out5 = self.bkbone(x)
    out5 = self.layer5(out5)
    out4 = self.layer4(out4)
    out3 = self.layer3(out3)

    out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
    out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)
    pred = torch.cat([out5, out4*out5, out3*out4*out5], dim=1)
    pred = self.predict(pred)
```

Backbone의 high-level에서 생성된 feature map의 크기를 동일하게 바꿔준 뒤, shallow attention과 concat을 하여 최종 prediction을 진행하게 됩니다.


## Requirements
* numpy
* opencv-python
* albumentations
* pytorch
* torchvision
* torchaudio
* pytorch-cuda


## How to run
### Run Train
    python train.py
#### train.py의 arguments
* datapath : 데이터 위치
* savepath : 모델.pth를 저장할 위치
* lr : learning rate
* epoch : train epoch수
* batch size : batch size
* weight_decay, momentum, nesterov : optimizer의 parameter
* num_workers : 데이터 로더의 worker수
* snapshot : pre-trained된 모델을 사용하여 학습할 경우 모델의 경로

---

### Run Test
    python test.py 
#### test.py의 arguments
* datapaths : 데이터 위치
* predpaths : prediction mask를 저장할 위치
* num_workers : 데이터 로더의 worker수
* snapshot : train된 모델의 경로


### Download dataset
[Train Dataset](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view)

[Test Dataset](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view)
