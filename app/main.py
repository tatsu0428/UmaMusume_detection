#必要なライブラリをインポート
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


#ロードするモデルの定義
class CNN(nn.Module):

  def __init__(self):
    super(CNN, self).__init__()
    self.cn1 = nn.Conv2d(3, 6, 5)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.cn2 = nn.Conv2d(6, 16, 5)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.cn3 = nn.Conv2d(16, 32, 4)
    self.dropout = nn.Dropout2d()
    self.fc1 = nn.Linear(32*10*10, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 5)
  
  def forward(self, x):
    x = F.relu(self.cn1(x))
    x = self.pool1(x)
    x = F.relu(self.cn2(x))
    x = self.pool2(x)
    x = F.relu(self.cn3(x))
    x = self.dropout(x)
    x = x.view(-1, 32*10*10)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
        
    return x


#読み込んだ画像の中からウマ娘の顔を検出し，名前とBoxを描画する関数
def detect(image, model):

    #顔検出器の準備
    classifier = cv2.CascadeClassifier("lbpcascade_animeface.xml")
    #画像をグレースケール化
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #画像の中から顔を検出
    faces = classifier.detectMultiScale(gray_image)

    #1人以上の顔を検出した場合
    if len(faces)>0:
        for face in faces:
            x, y, width, height = face
            detect_face = image[y:y+height, x:x+width]
            if detect_face.shape[0] < 64:
                continue
            detect_face = cv2.resize(detect_face, (64,64))
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            detect_face = transform(detect_face)
            detect_face = detect_face.view(1,3,64,64)

            output = model(detect_face)

            name_label = output.argmax(dim=1, keepdim=True)
            name = label_to_name(name_label)

            cv2.rectangle(image, (x,y), (x+width,y+height), (255, 0, 0), thickness=3) #四角形描画
            cv2.putText(image, name,(x,y+height+20), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),2) #人物名記述

    return image


#ラベルから対応するウマ娘の名前を返す関数
def label_to_name(name_label):

    if name_label == 0:
        name = "El Condor Pasa"
    elif name_label == 1:
        name = "Grass Wonder"
    elif name_label == 2:
        name = "King Halo"
    elif name_label == 3:
        name = "Seiun Sky"
    elif name_label == 4:
        name = "Special Week"
    
    return name


def main():

    st.set_page_config(layout="wide")
    #タイトルの表示
    st.title("ウマ娘 98世代 顔認識アプリ")
    #制作者の表示
    st.text("Created by Tatsuya NISHIZAWA")
    #アプリの説明の表示
    st.markdown("98世代のウマ娘の顔を識別するアプリです")

    #サイドバーの表示
    image = st.sidebar.file_uploader("画像をアップロードしてください", type=['jpg','jpeg', 'png'])
    #サンプル画像を使用する場合
    use_sample = st.sidebar.checkbox("サンプル画像を使用する")
    if use_sample:
        image = "sample.jpeg"

    #保存済みのモデルをロード
    model = CNN()
    model.load_state_dict(torch.load("cnn-99.model"))
    model.eval()

    #画像ファイルが読み込まれた後，顔認識を実行
    if image != None:
        
        #画像の読み込み
        image = np.array(Image.open(image))
        #画像からウマ娘の顔検出を行う
        detect_image = detect(image, model)
        #顔検出を行った結果を表示
        st.image(detect_image, use_column_width=True)


if __name__ == "__main__":
    #main関数の呼び出し
    main()
