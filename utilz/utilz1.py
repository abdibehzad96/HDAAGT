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
        self.Zoneindx = self.Nnodes - self.NZones
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


    def addnew(self, Scene, Target, Zones, ID, Fr):
        self.Scene = torch.cat((self.Scene, Scene.unsqueeze(0)), dim=0)
        # self.Adj_Mat_Scene = torch.cat((self.Adj_Mat_Scene, self.Adjacency(Scene, Zones[:self.sl], NObjs).unsqueeze(0)), dim=0)
        self.Zones.append(Zones) # = torch.cat((self.Zones, Zones), dim=0)
        self.ID.append(ID) #= torch.cat((self.ID, ID), dim=0)
        self.Fr.append(Fr) #= torch.cat((self.Fr, Fr), dim=0)
        # self.NUsers.append(NObjs) #= torch.cat((self.NUsers, torch.tensor(NObjs, device=self.device).unsqueeze(0)), dim=0)
        self.Target = torch.cat((self.Target, Target.unsqueeze(0)), dim=0)
        # self.Adj_Mat_Target = torch.cat((self.Adj_Mat_Target, self.Adjacency(Target,Zones[self.sl:], NObjs).unsqueeze(0)), dim=0)
        
    def Concat(self, fullscene, Zones, IDs, Fr, trORtst):
        # fullscene= self.speedcalc(fullscene)
        for j in range(0, self.sn*self.sw, self.sw):
            final_indx = j + self.sl + self.future
            start_indx = j + self.sl
            self.addnew(fullscene[j : j+ self.sl], fullscene[start_indx : final_indx], 
                        Zones[j : final_indx], IDs[j:final_indx],
                        Fr[j:final_indx])
        

    def augmentation(self, num):
        len = self.Scene.size(0)
        extenstion = torch.arange(self.Nnodes - self.NZones, self.Nnodes, device=self.device)
        for i in range(num):
            # Now we need to permute the rows of the Scene and the Target based on the order
            for n in range(len):
                order0 = torch.randperm(self.Nnodes - self.NZones,device=self.device)
                order = torch.cat((order0, extenstion), dim=0)
                newscene = self.Scene[n]
                zones = [z[order0] for z in self.Zones[n]]
                newtarget = self.Target[n]
                newscene = newscene[:,order]
                newtarget = newtarget[:,order]
                self.addnew(newscene, newtarget, zones, [], self.Fr[n], self.NUsers[n])
                

    def load_class(self, path, cmnt ):
        self.Scene = torch.load(os.path.join(path, cmnt, 'Scene.pt')) #, weights_only=True)
        self.Zones = torch.load(os.path.join(path, cmnt, 'Zones.pt'))#, weights_only=True)
        self.ID = torch.load(os.path.join(path, cmnt, 'ID.pt'))#, weights_only=True)
        self.Fr = torch.load(os.path.join(path, cmnt, 'Fr.pt'))#, weights_only=True)
        self.Adj_Mat_Scene = torch.load(os.path.join(path, cmnt, 'Adj_Mat_Scene.pt'))#, weights_only=True).to(self.device)
        self.NUsers = torch.load(os.path.join(path, cmnt, 'NUsers.pt')) #, weights_only=True)
        self.Target = torch.load(os.path.join(path, cmnt, 'Target.pt')) #, weights_only=True)
        self.Adj_Mat_Target = torch.load(os.path.join(path, cmnt, 'Adj_Mat_Target.pt')) #, weights_only=True).to(self.device)
        if self.trORtst == 0:
            self.maxval = torch.load(os.path.join(path, cmnt, 'maxval.pt'))
            self.minval = torch.load(os.path.join(path, cmnt, 'minval.pt'))

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
        return self.Scene[idx], self.Target[idx]
    
    


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
    xyid = args.xyidx
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
        torch.save(indices, os.path.join(os.getcwd(),'Pickled',f'indices.pt'))

    Dframe = torch.tensor(Dframe.values, dtype=torch.float, device=device)
    DFindx, DFindxtmp = 0, 0
    IDs = []
    Zones = []
    Fr = []
    rows = []
    TrfL = []
    for fr in range(int(min(Dframe[:,0])), int(max(Dframe[:,0]))):
        tracker = torch.zeros(Nnodes, device=device) + 4
        DFindxtmp = DFindx
        while Dframe[DFindx,0] == fr:
            DFindx += 1

        rows.append(Dframe[DFindxtmp:DFindx, Columns_to_keep]) # We only keep the columns that we need, for now Tr1, Tr2, Tr3, Tr4 are not used
        Fr.append(fr) # = torch.cat((Fr, fr), dim=0) # We keep the track of the frame number
        IDs.append(Dframe[DFindxtmp:DFindx, 1])
        Zones.append(Dframe[DFindxtmp:DFindx, 11])
        TrfL.append(Dframe[DFindx, TrfL_Columns])

        if len(rows) % scenelet_len == 0: # This only happens after each scenelet_len frames
            trORtst = checktstvstr(fr, indices, scenelet_len, train_size, test_size) # train, test, or validation
            # Scene, Zones, IDs = ConsistensyCheck2(Scene, Zones, IDs, Nnodes, NFeatures)
            Scene = ZoneConsistency(rows, Zones,TrfL, IDs, Nnodes, NFeatures, NZones, xyid, scenelet_len).to(device)
            if trORtst == 1: # test
                # .Concat(Scene, Zones, IDs, Fr, sn, sw, sl, framelen)
                Scenetst.Concat(Scene, Zones, IDs, Fr, trORtst)
            if not only_test: # 
                if trORtst == 0:
                    Scenetr.Concat(Scene, Zones, IDs, Fr,trORtst)
                elif trORtst == 2:
                    Sceneval.Concat(Scene, Zones, IDs, Fr,trORtst)
            Fr = [] #torch.empty(0, device=device)
            IDs = []
            Zones = []
            rows = []
    return Scenetr, Scenetst, Sceneval


def givindx(globlist, id):
    for i in range(len(globlist)):
        if globlist[i] == id:
            return i
    return None


def ZoneConsistency(rows, Zones, TrfL, IDs, Nnodes, NFeatures, NZones,xyid, scenelet_len): # Makes sure that the rows related to the same ID are in the same order in all the lists
    
    Scene = torch.zeros(scenelet_len, NZones-1, NFeatures)
    Zone_glob_ID_list = {} # we must create a global list for each zone, indipedent of the time steps
    tracker = 4 # indipedent of the time steps, depending on the zone
    for z in range(NZones-1):
        for t, Z_stp in enumerate(Zones): # now we start to sweep the zone through time steps
            Scene[t,z,:4] = TrfL[t] # We add the TrfL to the zone "z"
            lst = [Z_stp ==z+1] # ignore the zone 0
            ids = IDs[t][lst].tolist()
            # check if the lst is in the global list or else add it
            for n, id in enumerate(ids):
                if id not in Zone_glob_ID_list:
                    if tracker < NFeatures:
                        Zone_glob_ID_list[id] = tracker
                        Scene[t,z,Zone_glob_ID_list[id]:Zone_glob_ID_list[id]+2] = rows[t][lst][n][xyid]
                        tracker += 2
                else:
                    Scene[t,z,Zone_glob_ID_list[id]:Zone_glob_ID_list[id]+2] = rows[t][lst][n][xyid]
    
    return Scene




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

def Zoneconf(path = 'ZoneConf.yaml'):
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

if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")