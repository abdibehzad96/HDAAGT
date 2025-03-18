# This file will generate dataset for Scene centric models
from torch.utils.data import Dataset, DataLoader
import csv
import datetime
import os
import random
import math
import torch
import pandas as pd
import numpy as np
import yaml
import re
from shapely.geometry import Point, Polygon

class Scenes(Dataset):
    def __init__(self, args,trORtst): # Features include ['BBX','BBY','W', 'L' , 'Cls', 'Xreal', 'Yreal']
        self.Nnodes = args.Nnodes
        self.NFeatures = args.Nfeatures # initial number of features which the Vx, Vy and heading are added
        self.input_size = args.input_size # fullscene shape
        self.NZones = args.NZones
        self.sl = args.sl
        self.shift = args.sw
        self.sw = args.sw
        self.sn = args.sn
        self.future = args.future
        self.device = args.device
        self.framelen = 2*self.sl # number of frames used for training
        self.prohibitedZones = [200, 100] # These are the zones that are not allowed to be in the scene
        self.xyidx = args.xyidx
        self.eyemat = torch.eye(self.Nnodes, self.Nnodes, device=self.device)
        self.epsilon = -1e-16
        self.trORtst = trORtst
        self.Scene = torch.empty(0, self.sl, self.Nnodes, self.input_size, dtype=torch.long, device=self.device) #[Sampeles, Sequence Length, Nnodes (30), NFeatures(tba)]
        self.ID = [] #torch.empty(0, sl+future, Nnodes, device=device)
        self.Fr = [] #torch.empty(0, sl+future, device=device)
        self.Zones = [] #torch.empty(0, sl+future, Nnodes, device=device)
        self.NUsers = [] # torch.empty(0, device=self.device)
        self.Adj_Mat_Scene = torch.empty(0, self.sl, self.Nnodes, self.Nnodes, device=self.device)
        # self.Edge_Index = []
        self.Target = torch.empty(0, self.future, self.Nnodes, self.input_size, device=self.device, dtype= torch.long) #[Sampeles, Sequence Length, Nnodes (30), NFeatures(tba)]
        self.Adj_Mat_Target = torch.empty(0, self.future, self.Nnodes, self.Nnodes, device=self.device)
        self.Centre = args.Centre
        self.maxval = torch.zeros(self.input_size, device=self.device)
        self.minval = torch.zeros(self.input_size, device=self.device)


    def addnew(self, Scene, Target, Zones, ID, Fr, NObjs):
        self.Scene = torch.cat((self.Scene, Scene.unsqueeze(0)), dim=0)
        self.Adj_Mat_Scene = torch.cat((self.Adj_Mat_Scene, self.Adjacency(Scene, Zones[:self.sl], NObjs).unsqueeze(0)), dim=0)
        self.Zones.append(Zones) # = torch.cat((self.Zones, Zones), dim=0)
        self.ID.append(ID) #= torch.cat((self.ID, ID), dim=0)
        self.Fr.append(Fr) #= torch.cat((self.Fr, Fr), dim=0)
        self.NUsers.append(NObjs) #= torch.cat((self.NUsers, torch.tensor(NObjs, device=self.device).unsqueeze(0)), dim=0)
        self.Target = torch.cat((self.Target, Target.unsqueeze(0)), dim=0)
        # self.Adj_Mat_Target = torch.cat((self.Adj_Mat_Target, self.Adjacency(Target,Zones[self.sl:], NObjs).unsqueeze(0)), dim=0)
        
    def Concat(self, fullscene, Zones, IDs, Fr, NObjs, trORtst):
        fullscene= self.speedcalc(fullscene)
        for j in range(0, self.sn*self.sw, self.sw):
            final_indx = j + self.sl + self.future
            start_indx = j + self.sl
            scene = fullscene[j : j+ self.sl]
            ct = (scene[:,:,self.xyidx[0]] == 0).sum(dim=0) # count the number of zeros in the x direction
            ct = (ct < 10).view(1, self.Nnodes, 1) # if the number of zeros in the x direction is more than 10, we ignore the scene
            target = fullscene[start_indx : final_indx] * ct
            scene = scene* ct
            self.addnew(scene, target, 
                        Zones[j : final_indx], IDs[j:final_indx],
                        Fr[j:final_indx], NObjs)
        

    def augmentation(self, num): # it works with displacing the order of the users in the scene 
        len = self.Scene.size(0)
        extenstion = torch.arange(self.NUsers, self.Nnodes, device=self.device)
        for i in range(num):
            # Now we need to permute the rows of the Scene and the Target based on the order
            for n in range(len):
                order0 = torch.randperm(self.NUsers,device=self.device)
                order = torch.cat((order0, extenstion), dim=0)
                newscene = self.Scene[n]
                zones = [z[order0] for z in self.Zones[n]]
                newtarget = self.Target[n]
                newscene = newscene[:,order]
                newtarget = newtarget[:,order]
                self.addnew(newscene, newtarget, zones, [], self.Fr[n], self.NUsers[n])
        
    def addnoise(self, multiply, mx, ratio):
        ss = self.Scene.size()
        ts = self.Target.size()
        init_scene = self.Scene.clone()
        init_Adj_Mat_Scene = self.Adj_Mat_Scene.clone()
        init_Target = self.Target.clone()
        mask = torch.zeros_like(init_scene)
        mask[:,:,:, self.xyidx] = 1
        mask = mask * (init_scene!= 0)

        trmask = torch.zeros_like(init_Target)
        trmask[:,:,:, self.xyidx] = 1
        trmask = trmask * (init_Target!= 0)

        for i in range(multiply):
            noise = torch.randint(0, mx, ss, device=self.device)
            rand = torch.rand(ss, device=self.device) > ratio
            noisy_scene = init_scene + noise * rand * mask
            noise = torch.randint(0, mx, ts, device=self.device)
            rand = torch.rand(ts, device=self.device) > ratio
            noisy_tar = init_Target + noise * rand * trmask
            self.Scene = torch.cat((self.Scene, noisy_scene), dim=0)
            self.Adj_Mat_Scene = torch.cat((self.Adj_Mat_Scene, init_Adj_Mat_Scene), dim=0)
            self.Target = torch.cat((self.Target, noisy_tar), dim=0)



    def speedcalc(self, fullscene):
        x,y = fullscene[:, :, self.xyidx[0]].unsqueeze(2), fullscene[:, :, self.xyidx[1]].unsqueeze(2)
        dx = torch.diff(x, n=1, append=x[-1:,:], dim = 0)
        dy = torch.diff(y, n=1, append=y[-1:,:], dim = 0)
        flg = y!=0
        xc = ((x - self.Centre[0])/self.Centre[0])*flg
        yc = ((y - self.Centre[1])/self.Centre[1])*flg
        xc2 = xc**2
        yc2 = yc**2
        Rc = torch.sqrt(xc**2 + yc**2)*flg
        Rc2 = Rc**2
        # Rcinv = torch.nan_to_num(1/Rc, nan=0) * (y!=0)
        heading = torch.atan2(xc, yc)*flg
        SinX = torch.sin(xc)*flg
        CosY = torch.cos(yc)*flg
        SinY = torch.sin(yc)*flg
        CosX = torch.cos(xc)*flg
        Sin2X = torch.sin(2*xc)*flg
        Cos2X = torch.cos(2*xc)*flg
        Sin2Y = torch.sin(2*yc)*flg
        Cos2Y = torch.cos(2*yc)*flg


        fullscene = torch.cat((fullscene, dx, dy, heading, xc,yc, Rc, xc2, yc2, Rc2,
                                SinX, CosX, SinY, CosY,
                                Sin2X, Cos2X, Sin2Y, Cos2Y), dim=2)
        max,_ = fullscene.max(0)
        max, _ = max.max(0)
        min, _ = fullscene.min(0)
        min, _ = min.min(0)
        self.maxval = (self.maxval >= max)* self.maxval + (self.maxval < max)* max # find the maximum value in each feature to normalize the data
        self.minval = (self.minval <= min)* self.minval + (self.minval > min)* min # find the minimum value in each feature to normalize the data
        return fullscene


    def intention(self, dx, dy, heading):
        Intention = torch.zeros_like(dx)
        for i in range(dx.size(0)):
            for j in range(dx.size(1)):
                if heading[i,j] > 0.5 and heading[i,j] < 1.5:
                    Intention[i,j] = 1
                elif heading[i,j] > -1.5 and heading[i,j] < -0.5:
                    Intention[i,j] = 2
        return Intention
    def Adjacency(self, Scene, Zone, NObjs): # Features include ['W', 'L' , 'Cls', 'Xreal', 'Yreal']
        # Distance matrix
        Zone = torch.stack(Zone, dim=0)
        Adj_mat = torch.zeros(Scene.size(0), self.Nnodes, self.Nnodes, device=self.device) + self.epsilon
        Kmat0 = ((Zone[:, :NObjs]!=200) & (Zone[:, :NObjs]!=100)).float()
        Kmat = torch.mul(Kmat0.unsqueeze(2), Kmat0.unsqueeze(1))
        diff = Scene[:, :NObjs, self.xyidx].unsqueeze(2) - Scene[:, :NObjs, self.xyidx].unsqueeze(1)
        # Invsqr_dist = torch.sum(diff**2, dim=3).pow(-1).tanh().pow(0.5)
        Invsqr_dist = torch.sum((diff/1024)**2, dim=3).sqrt()
        Invsqr_dist = torch.exp(-Invsqr_dist)
        Adj_mat[:, :NObjs, :NObjs] = torch.mul(Invsqr_dist, Kmat)
        # Kmat = torch.mul(Kmat0, Zone[:,:NObjs] + self.Zoneindx) # Here we extend the Adj_mat to include the zones and connect Users to the zones
        # for k in range(NObjs):
        #     mask = Kmat[:, k] != 0 # Ignore the zones that are not in the scene over the frames( Sequence length)
        #     columns = Kmat[mask, k].int()
        #     rows = torch.nonzero(mask) # find the indices of the zones that are in the scene
        #     Adj_mat[rows, k , columns] = 1
        return Adj_mat
    
    def saturation(self, mat, maxval):
        mat = torch.relu(maxval - mat)
        mat = torch.relu(maxval - mat)
        return mat

    def load_class(self, path, cmnt ):
        self.Scene = torch.load(os.path.join(path, cmnt, 'Scene.pt')).to(self.device) #, weights_only=True)
        self.Zones = torch.load(os.path.join(path, cmnt, 'Zones.pt'))#, weights_only=True)
        self.ID = torch.load(os.path.join(path, cmnt, 'ID.pt'))#, weights_only=True)
        self.Fr = torch.load(os.path.join(path, cmnt, 'Fr.pt'))#, weights_only=True)
        self.Adj_Mat_Scene = torch.load(os.path.join(path, cmnt, 'Adj_Mat_Scene.pt')).to(self.device)#, weights_only=True).to(self.device)
        self.NUsers = torch.load(os.path.join(path, cmnt, 'NUsers.pt')) #, weights_only=True)
        self.Target = torch.load(os.path.join(path, cmnt, 'Target.pt')).to(self.device) #, weights_only=True)
        self.Adj_Mat_Target = torch.load(os.path.join(path, cmnt, 'Adj_Mat_Target.pt')).to(self.device) #, weights_only=True).to(self.device)
        if self.trORtst == 0:
            self.maxval = torch.load(os.path.join(path, cmnt, 'maxval.pt')).to(self.device)
            self.minval = torch.load(os.path.join(path, cmnt, 'minval.pt')).to(self.device)

    def save_class(self, path, cmnt):
        if not os.path.exists(os.path.join(path, cmnt)):
            os.makedirs(os.path.join(path, cmnt))
        else:
            #remove the existing files
            for file in os.listdir(os.path.join(path, cmnt)):
                os.remove(os.path.join(path, cmnt, file))
        torch.save(self.Scene, os.path.join(path, cmnt, 'Scene.pt'))
        torch.save(self.Zones, os.path.join(path, cmnt, 'Zones.pt'))
        torch.save(self.ID, os.path.join(path, cmnt, 'ID.pt'))
        torch.save(self.Fr, os.path.join(path, cmnt, 'Fr.pt'))
        torch.save(self.Adj_Mat_Scene, os.path.join(path, cmnt, 'Adj_Mat_Scene.pt'))
        torch.save(self.NUsers, os.path.join(path, cmnt, 'NUsers.pt'))
        torch.save(self.Target, os.path.join(path, cmnt,  f'Target.pt'))
        torch.save(self.Adj_Mat_Target, os.path.join(path, cmnt, 'Adj_Mat_Target.pt'))
        if self.trORtst == 0:
            torch.save(self.maxval, os.path.join(path, cmnt, 'maxval.pt'))
            torch.save(self.minval, os.path.join(path, cmnt, 'minval.pt'))

    def __len__(self):
        return self.Scene.size(0)
    
    def __getitem__(self, idx):
        return self.Scene[idx], self.Target[idx], self.Adj_Mat_Scene[idx] #, self.Adj_Mat_Target[idx]
    
    


# Headers = ['Frame', 'ID', 'BBx', 'BBy','W', 'L' , 'Cls','Tr1', 'Tr2', 'Tr3', 'Tr4', 'Zone', 'Xreal', 'Yreal']
def def_class(Scenetr, Scenetst, Sceneval, Dframe, args): # Nnodes, NFeatures, sl, future, sw, sn, Columns_to_keep, seed, ct, only_test, device):
    Nnodes = args.Nnodes
    NFeatures = args.Nfeatures
    NZones = args.NZones
    sl = args.sl
    future = args.future
    shift = args.sw
    sw = args.sw
    sn = args.sn
    Columns_to_keep = args.Columns_to_keep
    TrfL_Columns = args.TrfL_Columns
    NTrfL = len(TrfL_Columns)
    Seed = args.Seed
    ct = args.ct
    only_test = args.only_test
    device = args.device
    Nusers = args.Nusers
    
    scenelet_len = sl + future + sn*sw # number of frames in a scenelet
    scenelets = int((max(Dframe['Frame']) - min(Dframe['Frame']) + 1) // scenelet_len)
    train_size = int(0.9 * scenelets)
    test_size = int(0.1 * scenelets)
    val_size = scenelets - train_size - test_size
    savelog(f"Train size: {train_size}, Test size: {test_size}, Validation size: {val_size} ", ct)


    if Seed:
        savelog("Using seed for random split", ct)
        indices = torch.load(os.path.join(os.getcwd(),'Pickled',f'indices.pt'))
    else:
        savelog("Randomly splitting the data", ct)
        indices = list(range(scenelets))
        random.shuffle(indices)
        

    Dframe = torch.tensor(Dframe.values, dtype=torch.float, device=device)
    DFindx, DFindxtmp = 0, 0
    IDs = []
    Zones = []
    Fr = []
    Scene = torch.empty(0, Nnodes, NFeatures, device=device)
    TrfL_Extender = torch.zeros(NFeatures - NTrfL, device=device)
    for fr in range(int(min(Dframe[:,0])), int(max(Dframe[:,0]))):
        DFindxtmp = DFindx
        rows = torch.empty(0, NFeatures, device=device) # Nnodes and NFeatures are predefined fixed values, here is 30
        while Dframe[DFindx,0] == fr:
            DFindx += 1

        rows = Dframe[DFindxtmp:DFindx, Columns_to_keep] # We only keep the columns that we need, for now Tr1, Tr2, Tr3, Tr4 are not used
        Fr.append(fr) # = torch.cat((Fr, fr), dim=0) # We keep the track of the frame number
        IDs.append(Dframe[DFindxtmp:DFindx, 1])
        Zones.append(Dframe[DFindxtmp:DFindx, 11])
        TrfL = torch.cat((Dframe[DFindx, TrfL_Columns], TrfL_Extender), dim=0)
        while rows.size()[0] < Nusers:
            rows = torch.cat((rows, torch.zeros(1, NFeatures, device=device)), dim=0) # this is to make sure that the number of nodes is always Nnodes
        # for _ in range(NZones): # Traffic light states are a feature of the zones
        #     rows = torch.cat((rows, TrfL.unsqueeze(0)), dim=0)
            
        Scene = torch.cat((Scene, rows.unsqueeze(0)), dim=0) # One full scenelet after each scenelet_len frames

        if Scene.size(0) % scenelet_len == 0: # This only happens after each scenelet_len frames
            trORtst = checktstvstr(fr, indices, scenelet_len, train_size, test_size) # train, test, or validation
            # Scene, Zones, IDs = ConsistensyCheck2(Scene, Zones, IDs, Nnodes, NFeatures)
            Scene, Zones, IDs, NObjs = ConsistensyCheck2(Scene, Zones, IDs, Nnodes, NFeatures, Nusers)
            if trORtst == 1: # test
                # .Concat(Scene, Zones, IDs, Fr, sn, sw, sl, framelen)
                Scenetst.Concat(Scene, Zones, IDs, Fr, NObjs, trORtst)
            if not only_test: # 
                if trORtst == 0:
                    Scenetr.Concat(Scene, Zones, IDs, Fr, NObjs,trORtst)
                elif trORtst == 2:
                    Sceneval.Concat(Scene, Zones, IDs, Fr, NObjs,trORtst)
            Scene = torch.empty(0, Nnodes, NFeatures, device=device)
            Fr = [] #torch.empty(0, device=device)
            IDs = []
            Zones = []
    torch.save(indices, os.path.join(os.getcwd(),'Pickled',f'indices.pt'))
    return Scenetr, Scenetst, Sceneval, Scenetr.maxval, Scenetr.minval


def givindx(globlist, id):
    for i in range(len(globlist)):
        if globlist[i] == id:
            return i
    return None


def ConsistensyCheck2(Scene, Zones, IDs, Nnodes, NFeatures, Nusers): # Makes sure that the rows related to the same ID are in the same order in all the lists
    globlist = IDs[0] # Base list to compare the rest of the lists
    k = 0
    device = Scene.device
    SortedScene = torch.empty(0, Nnodes, NFeatures, device=device)
    SortedZones = []
    SortedIDs = []

    for ids in IDs[1:]: # We create a global list of all the IDs in the order they appear
        for id in ids:
            if id not in globlist:
                globlist= torch.cat((globlist,id.unsqueeze(0)), dim=0)

    for ids in IDs:
        Sc = torch.zeros(Nnodes, NFeatures, device=device)
        indxlist = torch.zeros(Nusers, device=device)
        Zonelist = 200*torch.ones(Nusers, device=device)
        for n , id in enumerate(ids): # We sweep through the idtmp list and compare it with the next frame's id list
            indx = givindx(globlist, id)
            if indx is None:
                print("Error in ConsistensyCheck2: indx is None")
            try:
                indxlist[indx] = id
                flg = 1
            except:
                print(f"Error in ConsistensyCheck2: Indx {indx} is out of range")
                flg = 0
            if flg:
                Sc[indx] = Scene[k, n] 
                Zonelist[indx] = Zones[k][n]

        SortedIDs.append(indxlist) # Sort the IDs based on the indxlist
        SortedZones.append(Zonelist) # Sort the Zones based on the indxlist
        SortedScene = torch.cat((SortedScene, Sc.unsqueeze(0)), dim=0)
        k += 1
    NObjs = len(globlist) if len(globlist) < Nusers else Nusers
    return SortedScene, SortedZones, SortedIDs, NObjs



def checktstvstr(fr, indices, scenelet_len, train_size, test_size):
    for indx , i in enumerate(indices):
        if i == fr//scenelet_len-1:
            break
    if indx <= train_size:
        return 0 # train
    elif indx > train_size and indx <= train_size + test_size:
        return 1 # test
    else:
        return 2 # validation

def prep_data(datasettr, datasettst, datasetval, batch_size):
    train_loader = DataLoader(datasettr, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasettst, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(datasetval, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader


def loadcsv(frmpath, Header, trjpath = None):
    df = pd.read_csv(frmpath, dtype =float)
    df.columns = Header
    if trjpath is not None:
        trj = pd.read_csv(trjpath, dtype =float)
        return df, trj
    # with open(csvpath, 'r',newline='') as file:
    #     for line in file:
    #         row = line.strip().split(',')
    #         # rowf = [float(element) for element in row]
    #         # rowf = [0 if math.isnan(x) else x for x in rowf]
    #         df.append(row)
    return df


def savecsvresult(pred , groundx, groundy):
    cwd = os.getcwd()
    ct = datetime.datetime.now().strftime(r"%m%dT%H%M")
    csvpath = os.path.join(cwd,'Processed',f'Predicteddata{ct}.csv')
    with open(csvpath, 'w',newline='') as file:
        writer = csv.writer(file)
        for i in range(len(groundx)):
            rowx = pred[i,:,0].tolist()
            rowy = pred[i,:,1].tolist()
            grx = groundx[i,:,0].tolist()
            gry = groundx[i,:,1].tolist()
            grxx = groundy[i,:,0].tolist()
            gryy = groundy[i,:,1].tolist()
            writer.writerow([rowx,grxx,grx])
            writer.writerow([rowy,gryy,gry])




def savelog(log, ct): # append the log to the existing log file while keeping the old logs
    # if the log file does not exist, create one
    print(log)
    if not os.path.exists(os.path.join(os.getcwd(),'logs')):
        os.mkdir(os.path.join(os.getcwd(),'logs'))
    with open(os.path.join(os.getcwd(),'logs', f'log-{ct}.txt'), 'a') as file:
        file.write('\n' + log)
        file.close()

def Zoneconf(path = '/home/abdikhab/New_Idea_Traj_Pred/utilz/ZoneConf.yaml'):
    ZoneConf = []
    with open(path) as file:
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
    B, Nnodes,_ = BB.size()
    BB = BB.reshape(-1,2).cpu()
    PredZone = torch.zeros(B*Nnodes, device=BB.device)
    for n , bb in enumerate(BB.int()):
        for i, zone in enumerate(Zones):
            Poly = Polygon(zone)
            if Poly.contains(Point(bb[0], bb[1])):
                PredZone[n] = i+1
                break
        
    
    return PredZone.reshape(B, Nnodes)

def Zone_compare(Pred, Target, PrevZone, BB):
    # possiblemoves = torch.tensor([0,2,5,7,8],[0,1,2,7,8],[2],[0,2,3,5,8],[0,2,4,5,7], [5],[0,5,6,7,8],[7],[8])
    singlezones = torch.tensor([3,6,8,9], device=Pred.device)
    neighbours = torch.tensor([[0],[1],[6],[7],[8],[9],[2],[3],[4],[5]], device=Pred.device)
    B, Nnodes = Pred.size()
    Pred = Pred.reshape(-1)
    Target = Target.reshape(-1).cpu()
    PrevZone = PrevZone.reshape(-1).cpu()
    totallen = B*Nnodes
    count = 0
    nonzero = 0
    doublezone = 0
    for i in range(B*Nnodes):
        if Target[i] != 0:
            nonzero += 1
            if Pred[i] == Target[i]:
                    count += 1
            else:
                if PrevZone[i] in singlezones:
                    totallen -= 1
                    # print("Single Zone")
                if Pred[i]== neighbours[Target[i].int()]:
                    doublezone += 1
                
    return count, totallen, B*Nnodes, nonzero, doublezone


def target_mask(trgt, num_head = 0, device="cuda:3"):
    B, SL, Nnodes, _= trgt.size()
    mask = 1- torch.triu(torch.ones(SL, SL, device=trgt.device), diagonal=1)# Upper triangular matrix
    return mask == 0

def target_mask0(trgt, num_head, device="cuda:3"):
    B, SL, Nnodes, _= trgt.size()
    mask = 1- torch.triu(torch.ones(SL, SL, device=trgt.device), diagonal=1).unsqueeze(0).unsqueeze(0)  # Upper triangular matrix
    mask = mask.unsqueeze(4) * (trgt[:,:,:,1] != 0).unsqueeze(1).unsqueeze(3)
    if num_head > 1:
        mask = mask.repeat_interleave(num_head,dim=0)
    return mask == 0

def create_src_mask(src, device="cuda:3"):
    mask = src[:,:,:, 1] == 0
    return mask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")