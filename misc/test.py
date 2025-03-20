import argparse
import os
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import datetime


learning_rate, hidden_size, num_layers, batch_size, ct= 5e-4, 128, 2, 128, '11-19-15-08'
code = f"LR{int(learning_rate*100000)}_HS{hidden_size}_NL{num_layers}_BS{batch_size} {ct}"
print(f"Code: {code}")
Predicteddata  = torch.load(os.path.join(os.getcwd(),'Pickled', code + 'Predicteddata.pt'), weights_only=True)
ActualTarget = torch.load(os.path.join(os.getcwd(),'Pickled', 'Test', 'Target.pt'), weights_only=True)
ActualScene = torch.load(os.path.join(os.getcwd(),'Pickled', 'Test', 'Scene.pt'), weights_only=True)
Adj_Mat_Scene = torch.load(os.path.join(os.getcwd(),'Pickled', 'Test', 'Adj_Mat_Scene.pt'), weights_only=True)
Adj_Mat_Target = torch.load(os.path.join(os.getcwd(),'Pickled', 'Test', 'Adj_Mat_Target.pt'), weights_only=True)
test_loss = torch.load(os.path.join(os.getcwd(),'Pickled', code + 'test_losses.pt'), weights_only=True)
epoch_losses = torch.load(os.path.join(os.getcwd(),'Pickled', code + 'epoch_losses.pt'), weights_only=True)

from utilz.utils import *
import torch.nn.functional as F
from models.HGAAGT import *
Best_Model=torch.load(os.path.join(os.getcwd(),'Processed', code + 'Bestmodel.pth'))
# Best_Model_dict = Best_Model.state_dict()

# Model = GGAT(args)
# Model.load_state_dict(Best_Model_dict)

def eval(model, Scene, Adj_mat, h0, Nseq):
    # savelog("Starting testing phase", ct)
    model.eval()
    with torch.no_grad():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        Out, Gruh0 = model(Scene, Adj_mat, h0, Nseq)
        end_event.record()
        torch.cuda.synchronize()
        inference_time = start_event.elapsed_time(end_event)
    return Out, Gruh0, inference_time

def attach(NScene, Scenelet, Zonecft):
    x,y = NScene[:,-11:,:,0].unsqueeze(3), NScene[:,-11:,:,1].unsqueeze(3)

    dx = torch.diff(x, n=1, dim = 1)
    dy = torch.diff(y, n=1, dim = 1)
    heading = torch.atan2(dy, dx)
    xc = x[:,-10:] - 512
    yc = y[:,-10:] - 512
    Rc = torch.sqrt(xc**2 + yc**2)
    SinX = torch.sin(xc/512)
    CosY = torch.cos(yc/512)
    SinY = torch.sin(yc/512)
    CosX = torch.cos(xc/512)
    Sin2X = torch.sin(2*xc/512)
    Cos2X = torch.cos(2*xc/512)
    Sin2Y = torch.sin(2*yc/512)
    Cos2Y = torch.cos(2*yc/512)


    added = torch.cat((NScene[:,-10:,:,:2], Scenelet, dx, dy, heading, xc,yc, Rc,
                            SinX, CosX, SinY, CosY,
                            Sin2X, Cos2X, Sin2Y, Cos2Y), dim=3)
    NScene[:,-10:] = added
    for i in range(10):
        for j in range(NScene.size(2)):
            NScene[:,i+10,j,9] = torch.tensor(zonefinder(NScene[0,i+10,j,:2].cpu().numpy(), Zonecft), device=NScene.device)
    return NScene

def Zoneconf():
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
    return ZoneConf

def zonefinder(BB, Zones):
    for i, zone in enumerate(Zones):
        Poly = Polygon(zone)
        if Poly.contains(Point(BB[0], BB[1])):
            return i+1
    return 100

def scene_eval(Best_Model, Scene, Adj_Mat,Target, future, Zonecft):
    Pred = []
    total_time = 0
    NScene, GRUh0, inference_time = eval(Best_Model, Scene[:,-20:,:,:], Adj_Mat[:,:20,:32,:32], None, False)
    Pred.append(NScene[:,-1].unsqueeze(0))
    NScene = attach(NScene[:,-1], Target[:,0,:,2:10], Zonecft).unsqueeze(0)
    
    for f in range(future-1):
        NScene1, GRUh0, inference_time = eval(Best_Model, NScene, Adj_Mat[:,f+1,:32,:32].unsqueeze(1), GRUh0[:,:,-1,:].unsqueeze(2), True)
        Pred.append(NScene1[:,-1].unsqueeze(0))
        NScene = attach(NScene1[:,-1], Target[:,f+1,:,2:10], Zonecft).unsqueeze(0)
        total_time += inference_time

    Pred = torch.cat(Pred, dim=1).to(torch.int)
    tag = Target[:,:,:,:2] != 0
    Pred = Pred*(tag.to(torch.int))

    loss = F.mse_loss(Pred, Target[:,:,:,:2])
    msePtime = torch.sqrt(torch.sum((Pred - Target[:,:,:,:2]) ** 2, dim = -1)).sum(dim = -1)/32
    msePuser = torch.sqrt(torch.sum((Pred - Target[:,:,:,:2]) ** 2, dim = -1)).sum(dim = 1)/future

    # print(f"loss: {loss.item()} and inference time: {total_time} ms")
    # print(f"mse per time step: {msePtime}")
    # print(f"mse per user: {msePuser}")
    return Pred, loss, msePtime.squeeze(0), msePuser.squeeze(0), total_time

def scene_eval20(Best_Model, Scene, Adj_Mat,Target, future, Zonecft):
    Pred = []
    total_time = 0
    NScene, GRUh0, inference_time = eval(Best_Model, Scene[:,-20:,:,:], Adj_Mat[:,1:21,:32,:32], None, False)
    Pred.append(NScene[:,-1].unsqueeze(0))
    NScene = attach(NScene[:,-1], Target[:,0,:,2:10], Zonecft).unsqueeze(0)
    NScene = torch.cat((Scene[:,-19:,:,:], NScene), dim=1)
    
    for f in range(future-1):
        NScene1, GRUh0, inference_time = eval(Best_Model, NScene, Adj_Mat[:,f+2:f+22,:32,:32], None, False)
        Pred.append(NScene1[:,-1].unsqueeze(0))
        NScene1 = attach(NScene1[:,-1], Target[:,f+1,:,2:10], Zonecft).unsqueeze(0)
        NScene = torch.cat((NScene[:,-19:,:,:], NScene1), dim=1)
        total_time += inference_time

    Pred = torch.cat(Pred, dim=1)
    tag = Target[:,:,:,:2] != 0
    Pred = Pred*(tag.to(torch.int))

    loss = F.mse_loss(Pred, Target[:,:,:,:2])
    msePtime = torch.sqrt(torch.sum((Pred - Target[0,:,:,:2]) ** 2, dim = -1)).sum(dim = -1)/32
    msePuser = torch.sqrt(torch.sum((Pred - Target[0,:,:,:2]) ** 2, dim = -1)).sum(dim = 1)/future

    # print(f"loss: {loss.item()} and inference time: {total_time} ms")
    # print(f"mse per time step: {msePtime}")
    # print(f"mse per user: {msePuser}")
    return Pred, loss, msePtime.squeeze(0), msePuser.squeeze(0), total_time


def scene_eval10(Best_Model, Scene, Adj_Mat,Target, future, Zonecft):
    Pred = []
    total_time = 0
    NScene, GRUh0, inference_time = eval(Best_Model, Scene, Adj_Mat[:,:20,:32,:32], None, False)
    Pred.append(NScene[:, :-10])
    NScene1 = torch.zeros(1, 10, 32, Scene.size(-1), device=Scene.device)
    NScene1[:,:,:,:2] = NScene[:,-10:]
    NScene = torch.cat((Scene[:,-10:,:,:], NScene1), dim=1)
    NScene = attach(NScene, Target[:,10:20,:,2:10], Zonecft)
    
    
    for f in range(2):
        NScene, GRUh0, inference_time = eval(Best_Model, NScene, Adj_Mat[:,(f+3)*10:(f+4)*10,:32,:32], None, False)
        Pred.append(NScene[:, :-10])
        NScene1 = torch.zeros(1, 10, 32, Scene.size(-1), device=Scene.device)
        NScene1[:,:,:,:2] = NScene[:,-10:]
        NScene = torch.cat((Scene[:,-10:,:,:], NScene1), dim=1)
        NScene = attach(NScene, Target[:,(f+2)*10:(f+3)*10,:,2:10], Zonecft)
        total_time += inference_time

    Pred = torch.cat(Pred, dim=1)
    tag = Target[:,10:,:,:2] != 0
    Pred = Pred*(tag.to(torch.int))

    loss = F.mse_loss(Pred, Target[:,10:,:,:2])
    msePtime = torch.sqrt(torch.sum((Pred - Target[0,:,:,:2]) ** 2, dim = -1)).sum(dim = -1)/32
    msePuser = torch.sqrt(torch.sum((Pred - Target[0,:,:,:2]) ** 2, dim = -1)).sum(dim = 1)/future

    # print(f"loss: {loss.item()} and inference time: {total_time} ms")
    # print(f"mse per time step: {msePtime}")
    # print(f"mse per user: {msePuser}")
    return Pred, loss, msePtime.squeeze(0), msePuser.squeeze(0), total_time

future = 30
Prediction = []
totalmsePtime = torch.zeros(30, device=Adj_Mat_Scene.device)
totalmsePuser = torch.zeros(32, device=Adj_Mat_Scene.device)
total_loss = 0

total_time = 0
Adj_Mat = torch.cat((Adj_Mat_Scene, Adj_Mat_Target), dim=1).unsqueeze(0)
Zonecft = Zoneconf()
for n in range(250):
    Scene = ActualScene[n,:,:32,:].unsqueeze(0)
    Target = ActualTarget[n,:,:32].unsqueeze(0)
    adj_mat = Adj_Mat[:,n,:,:32,:32]
    Pred, loss, msePtime, msePuser, elapsed_time = scene_eval10(Best_Model, Scene, adj_mat, Target, future, Zonecft)
    totalmsePtime = totalmsePtime+ msePtime
    totalmsePuser = totalmsePuser+ msePuser
    total_loss += loss
    total_time += elapsed_time
    Prediction.append(Pred)

totalmsePtime = totalmsePtime/250
totalmsePuser = totalmsePuser/250
total_loss = total_loss/250
Prediction = torch.cat(Prediction, dim=0)
print(f"Total loss: {total_loss.item()}")
print(f"Total mse per time step: {totalmsePtime}")
print(f"Total mse per user: {totalmsePuser}")
print(f"Total inference time: {total_time/250} ms")
print("done")