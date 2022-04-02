# ウマ娘　98世代　顔認識アプリ
98世代のウマ娘を検出するアプリです．

## 98世代とは
日本の競走馬で1995年生まれの世代を指します．

ウマ娘では**スペシャルウィーク**，**セイウンスカイ**，**キングヘイロー**，**エルコンドルパサー**，**グラスワンダー**の5人を指しています．

詳細：https://dic.pixiv.net/a/98%E4%B8%96%E4%BB%A3#h2_3

## 実行例

![localhost_8501_](https://user-images.githubusercontent.com/84188861/161369444-39eb1e13-e05d-4a35-a315-3de390fd5cc9.png)

## 実行方法

1. appフォルダ内の"main.py"を実行します．

```bash
streamlit run main.py
```

2. ブラウザ上にアプリケーションが立ち上がるので，そこにウマ娘の画像をアップロードすると顔認識を行います．

## 環境

* Python 3.8.3
* numpy 1.21.4
* opencv-python 4.5.5.64
* Pillow 7.2.0
* torch 1.9.0
* torchvision 0.11.3
* streamlit 1.5.1

CNNモデルの構築にPytorch，アプリケーションの開発にStreamlitを使用しています．

## モデルの構造

```bash
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 60, 60]             456
         MaxPool2d-2            [-1, 6, 30, 30]               0
            Conv2d-3           [-1, 16, 26, 26]           2,416
         MaxPool2d-4           [-1, 16, 13, 13]               0
            Conv2d-5           [-1, 32, 10, 10]           8,224
         Dropout2d-6           [-1, 32, 10, 10]               0
            Linear-7                  [-1, 120]         384,120
            Linear-8                   [-1, 84]          10,164
            Linear-9                    [-1, 5]             425
================================================================
Total params: 405,805
Trainable params: 405,805
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 0.36
Params size (MB): 1.55
Estimated Total Size (MB): 1.95
----------------------------------------------------------------
```





