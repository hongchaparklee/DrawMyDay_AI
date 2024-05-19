import cv2
import torch
import os

import torch.nn as nn
from torchvision import transforms 
from torchvision.datasets import ImageFolder
from PIL import Image
import requests
import ssl

#import torch.load with map_location=torch.device('cpu')

ssl._create_default_https_context = ssl._create_unverified_context

with open('./result.txt', 'w') as file:
    pass

class_names = []
f = open("labels.txt",'r')

lines = f.readlines()
for line in lines:
    line = line.strip()
    class_names.append(line)

trans = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomInvert(1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])])

def idx_to_chr(x) :
    x += 44032
    return chr(x)


img = cv2.imread('./recieve/paper.png')
print(img.shape)

count = 0
for i in range(9) :
    for j in range(12) :
        tmp = img[j*64+1:j*64+64,i*64+1 : i*64+64]
        cv2.imwrite(f'./recieve/words/word{count}.png',tmp)
        count+=1
        #print(tmp.shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from torchvision import models
#resnet = torch.load("./model/(best)resnet18_224.pt")

resnet = models.resnet18(pretrained=False)
num_ftrs = resnet.fc.in_features
print(num_ftrs)
resnet.fc = nn.Linear(num_ftrs, 2361) 
resnet.load_state_dict(torch.load("./model/model_weights.pth",map_location=device))
resnet.to(device)
resnet.eval()

out = ""
for k in range(count) :
    image = Image.open(f'./recieve/words/word{k}.png')  # 추론할 이미지 불러오기
    image = image.convert("RGB")
    image = trans(image)  # 이미지 변환
    
    cvimg = cv2.imread(f'./recieve/words/word{k}.png')
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
    m = cvimg.shape[0]
    n = cvimg.shape[1]

    isempty = True
    for i in range(m) :
        for j in range(n) :
            if cvimg[i][j] != 255 :
                isempty = False

    if isempty :
        out = out + " "
        continue

    with torch.no_grad():  # 추론 시에는 그라디언트를 계산할 필요 없음
        inputs = image.unsqueeze(0)  # 배치 차원 추가 (batch_size=1)
        inputs =inputs.to(device)
        outputs = resnet(inputs)  # 모델에 이미지 입력하여 추론 수행3
        _, predicted = torch.topk(outputs, 5)  # 예측된 클래스 인덱스

    
    for i in range(1):
        item = predicted[0][i].item()
        label = class_names[item]
        character = idx_to_chr(int(label[5:]))
        out = out + character
        #print("Predicted class:", character , "with probability:", torch.softmax(outputs, 1)[0][predicted[0][i]].item())


f=open('./result.txt','w')
f.write(out)

#os.system("rm -rf ./recieve/words/*")
#os.system("rm -f ./recieve/paper.png")


