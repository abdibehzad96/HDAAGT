#This file is used to plot the trajectories of the preprocessed data
import matplotlib.pyplot as plt
import os
import math
import cv2
import yaml
import re
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from utilz.utils import *
from models.encdec import *

import re
ZoneConf = []
with open('utilz/ZoneConf.yaml') as file:
    ZonesYML = yaml.load(file, Loader=yaml.FullLoader)
    #convert the string values to float
    for _, v in ZonesYML.items():
        lst = []
        for _, p  in v.items():    
            for x in p[0]:
                b = re.split(r'[,()]',p[0][x])
                lst.append((float(b[1]), float(b[2])))
        ZoneConf.append(lst)
from shapely.geometry import Point, Polygon
def zonefinder(BBx,BBy, Zones):
    for i, zone in enumerate(Zones):
        Poly = Polygon(zone)
        if Poly.contains(Point(BBx, BBy)):
            return i
    return 100

def Move(zn):
    movement = np.zeros((5,1))
    i = 0
    for n, inx in enumerate(zn[1:]):
        if inx != zn[n]:
            movement[inx] = 1
            movement[zn[n]] = 1
            i += 1
            print("moved")
    return i

def create_video(path = '/home/abdikhab/Traj_Pred/encdec/outputs'):
    print("Creating video")
    cwd = os.getcwd()
    video_path = os.path.join(path,"encdec_out.mp4")
    list = os.listdir(path)
    list.sort()
    img = cv2.imread("/home/abdikhab/Traj_Pred/RawData/test.jpg")
    height, width, _ = img.shape
    video = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'mp4v'),0.5,(width,height))
    for lst in list:
        images = cv2.imread(os.path.join(path, lst))
        video.write(images)
    video.release()
    print("Video created successfully")



def imgcreator(Predicteddata, ego, target, path = "/home/abdikhab/Traj_Pred/encdec/outputs"):
    print("Creating images")
    # green = (0,255,0)
    # red = (0,0,255)
    # blue = (255,0,0)
    # remove the directory and create a new one
    if os.path.exists(path):
        os.system(f"rm -r {path}")
    os.mkdir(path)
    n = 0
    for pred in Predicteddata:
        pred = pred.to('cpu').numpy()
        grth = ego[n].to('cpu').numpy()
        grth_target = target[n].to('cpu').numpy()
        img = cv2.imread("/home/abdikhab/Traj_Pred/RawData/test.jpg")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # add points to the coordinates on pred, grth and grth_target
        for i in range(len(pred)):
            cv2.circle(img, (int(pred[i,0]), int(pred[i,1])), 4, (0,255,0), -1)
            cv2.circle(img, (int(grth[i,0]), int(grth[i,1])), 4, (0,0,255), -1)
            cv2.circle(img, (int(grth_target[i,0]), int(grth_target[i,1])), 4, (255,0,0), -1)
        cv2.imwrite(os.path.join(path, f"img{n}.jpg"), img)
        n += 1
        #plot polygon on the image
        # for i, zone in enumerate(ZoneConf):
        #     cv2.polylines(img, [np.array(zone).astype(np.int32)], isClosed=True, color=(0,0,0), thickness=1)
        # plt.imshow(img)
        # plt.legend(['Predicted', 'Ego', 'Target'])
    print("Images created successfully")


input_size, hidden_size, num_layers, output_size, epochs, patience_limit,  learning_rate = 10, 512, 1, 2, 300, 25, 0.01
sl, sw , shift, batch_size = 20, 10, 20, 256
future = 10
ct = '05-15-13-40'
code = f"LR{int(learning_rate*1000)}_HS{hidden_size}_NL{num_layers}_BS{batch_size} {ct}"
cwd = os.getcwd()
csvpath = os.path.join(cwd,'Processed','Trj20240510T1529.csv')
model = torch.load(os.path.join(os.getcwd(),'Processed', code + 'Bestmodel.pth'))
criterion = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#     savelog(f"Using {torch.cuda.device_count()} GPUs", ct)
#     model = nn.DataParallel(model)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate )

df = loadcsv(csvpath)
dataset= Scene_Process(df, sw, sl, shift, future, input_size, output_size, tr2img= True, ct= ct, imgsize=1024, Dsample = 4, normalize = False, device = device)
train_loader, test_loader, val_loader = prep_data(dataset, batch_size, ct, seed = True) #seed must always be true


Predicteddata, target, ego, avg_test_loss, accuracy, log = test_model(model, test_loader, future, output_size, device, criterion)
print("Model tested")
imgcreator(Predicteddata, ego, target)
create_video()