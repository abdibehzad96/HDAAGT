# This file will generate dataset for Scene centric models
from torch.utils.data import Dataset, DataLoader
import csv
import datetime
import os
import random
import torch
import pandas as pd
import yaml
import re
from shapely.geometry import Point, Polygon

class Scenes(Dataset):
    def __init__(self, config): 
        self.Nnodes = config['Nnodes']
        self.NFeatures = config['NFeatures'] # initial number of features which the Vx, Vy and heading are added
        self.input_size = config['input_size'] # fullscene shape
        self.NZones = config['NZones']
        self.sl = config['sl']
        self.shift = config['sw']
        self.sw = config['sw']
        self.sn = config['sn']
        self.future = config['future']
        self.device = config['device']
        self.framelen = 2*self.sl # number of frames used for training
        self.prohibitedZones = [200, 100] # These are the zones that are not allowed to be in the scene, such as parking lots and sidewalks, etc. This zones will be ignored
        self.xyidx = config['xy_indx']
        self.Centre = config['Centre']
        self.dwn_smple = config['dwn_smple']
        self.eyemat = torch.eye(self.Nnodes, self.Nnodes, device=self.device)
        self.epsilon = -1e-16
        self.ID = [] # We keep the track of the object IDs
        self.Fr = [] # Same as ID but for the frame number
        self.Zones = []
        self.NUsers = [] # Number of users in each scene, this helps decreasing the preprocessings
        self.Scene = torch.empty(0, self.sl, self.Nnodes, self.input_size, dtype=torch.long, device=self.device) #[Sampeles, Sequence Length, Nnodes, NFeatures], Scene is the input to the model
        self.Adj_Mat_Scene = torch.empty(0, self.sl, self.Nnodes, self.Nnodes, device=self.device) #[Sampeles, Sequence Length, Nnodes, Nnodes], Adjacency matrix for the Scene
        self.Target = torch.empty(0, self.future, self.Nnodes, self.input_size, device=self.device, dtype= torch.long) #[Sampeles, Sequence Length, Nnodes, NFeatures], Target is the output of the model


    def addnew(self, Scene, Target, Zones, ID, Fr, NObjs):
        self.Scene = torch.cat((self.Scene, Scene.unsqueeze(0)), dim=0)
        self.Adj_Mat_Scene = torch.cat((self.Adj_Mat_Scene, self.Adjacency(Scene, Zones[:self.sl], NObjs).unsqueeze(0)), dim=0)
        self.Zones.append(Zones) 
        self.ID.append(ID)
        self.Fr.append(Fr)
        self.NUsers.append(NObjs)
        self.Target = torch.cat((self.Target, Target.unsqueeze(0)), dim=0)
        
    def Slide_(self, fullscene, Zones, IDs, Fr, NObjs): # Concatenate the 
        fullscene= self.ExtractFeatures(fullscene) # Extract more features from the fullscene
        for j in range(0, self.sn*self.sw, self.sw):
            final_indx = j + self.sl + self.future
            start_indx = j + self.sl
            scene = fullscene[j : j+ self.sl: self.dwn_smple]
            no_show_frames = (scene[:,:,self.xyidx[0]] == 0).sum(dim=0) # The number of zeros in the x direction, which means they are out of camera sight
            no_show_frames = (no_show_frames < self.sl//2//self.dwn_smple).view(1, self.Nnodes, 1) # At least the vehicle must be half of the sequence, unless we ignore the scene
            target = fullscene[start_indx : final_indx:self.dwn_smple] * no_show_frames
            scene = scene* no_show_frames
            self.addnew(scene, target, 
                        Zones[j : final_indx: self.dwn_smple], IDs[j:final_indx: self.dwn_smple],
                        Fr[j:final_indx: self.dwn_smple], NObjs)
        

    def augmentation(self, num): # it works with permuting the order of the users in the scene  "num" times
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
        
    def addnoise(self, multiply, mx, ratio): # this adds noise "multiply" times to the scene and the target, it works by mx amplitude and probability of ratio
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



    def ExtractFeatures(self, fullscene):
        x,y = fullscene[:, :, self.xyidx[0]].unsqueeze(2), fullscene[:, :, self.xyidx[1]].unsqueeze(2)
        dx = torch.diff(x, n=1, append=x[-1:,:], dim = 0)
        dy = torch.diff(y, n=1, append=y[-1:,:], dim = 0)
        flg = y!=0 # In most cases, all Nodes are not filled with agents, so we must ignore them
        xc = ((x - self.Centre[0])/self.Centre[0])*flg
        yc = ((y - self.Centre[1])/self.Centre[1])*flg
        xc2 = xc**2
        yc2 = yc**2
        Rc = torch.sqrt(xc**2 + yc**2)*flg
        Rc2 = Rc**2
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
        return fullscene


    def Adjacency(self, Scene, Zone, NObjs): # Create the adjacency matrix for the Scene agents, the values are the normalized inverse of the distance between the agents
        Adj_mat = torch.zeros(Scene.size(0), self.Nnodes, self.Nnodes, device=self.device) + self.epsilon
        Kmat0 = ((Zone[:, :NObjs]!=200) & (Zone[:, :NObjs]!=100)).float()
        Kmat = torch.mul(Kmat0.unsqueeze(2), Kmat0.unsqueeze(1))
        diff = Scene[:, :NObjs, self.xyidx].unsqueeze(2) - Scene[:, :NObjs, self.xyidx].unsqueeze(1)
        # Invsqr_dist = torch.sum(diff**2, dim=3).pow(-1).tanh().pow(0.5)
        Invsqr_dist = torch.sum((diff/1024)**2, dim=3).sqrt()
        Invsqr_dist = torch.exp(-Invsqr_dist)
        Adj_mat[:, :NObjs, :NObjs] = torch.mul(Invsqr_dist, Kmat)
        return Adj_mat


    def load_class(self, path, cmnt): # It usually takes time preparing the dataset, so it's better to save it and load it later
        self.Scene = torch.load(os.path.join(path, cmnt, 'Scene.pt'), weights_only=True).to(self.device)
        self.Zones = torch.load(os.path.join(path, cmnt, 'Zones.pt'), weights_only=True)
        self.ID = torch.load(os.path.join(path, cmnt, 'ID.pt'), weights_only=True)
        self.Fr = torch.load(os.path.join(path, cmnt, 'Fr.pt'), weights_only=True)
        self.Adj_Mat_Scene = torch.load(os.path.join(path, cmnt, 'Adj_Mat_Scene.pt'), weights_only=True).to(self.device)
        self.NUsers = torch.load(os.path.join(path, cmnt, 'NUsers.pt'), weights_only=True)
        self.Target = torch.load(os.path.join(path, cmnt, 'Target.pt'), weights_only=True)

    def save_class(self, path, cmnt):
        if not os.path.exists(os.path.join(path, cmnt)): # Create the directory if it doesn't exist
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
        torch.save(self.Target, os.path.join(path, cmnt,  'Target.pt'))

    def __len__(self):
        return self.Scene.size(0)
    
    def __getitem__(self, idx):
        return self.Scene[idx], self.Target[idx], self.Adj_Mat_Scene[idx] #, self.Adj_Mat_Target[idx] , self.ID[idx], self.Fr[idx], self.Zones[idx]
    
    

def Scene_Process(Scenetr, Scenetst, Sceneval, Traffic_data, config): # Nnodes, NFeatures, sl, future, sw, sn, Columns_to_keep, seed, ct, only_test, device):
    Nnodes = config['Nnodes']
    NFeatures = config['NFeatures']
    sl = config['sl']
    future = config['future']
    sw = config['sw']
    sn = config['sn']
    Columns_to_keep = config['Columns_to_keep']
    Seed = not config['generate_data']
    ct = config['ct']
    only_test = config['only_test']
    device = config['device']
    Nusers = config['Nusers']
    
    scenelet_len = sl + future + sn*sw # Maximum number of frames in a scenelet
    scenelets = int((max(Traffic_data['Frame']) - min(Traffic_data['Frame']) + 1) // scenelet_len)
    train_size = int(0.9 * scenelets)
    test_size = int(0.1 * scenelets)
    val_size = scenelets - train_size - test_size
    savelog(f"Train size: {train_size}, Test size: {test_size}, Validation size: {val_size} ", ct)
    if Seed:
        savelog("Using seed for random split", ct)
        indices = torch.load(os.path.join(os.getcwd(),'Pickled',f'indices.pt'))
    else:
        savelog("Randomly splitting the data", ct)
        indices = torch.randperm(scenelets) # To avoid data leakage, seed selects between the scenelet_len frames, so sliding window does not affect it
        

    Traffic_data = torch.tensor(Traffic_data.values, dtype=torch.float, device=device)
    DFindx, DFindxtmp = 0, 0 # We sweep through the first frame to the last frame
    IDs = []
    Zones = []
    Fr = []
    Scene = []
    for fr in range(int(min(Traffic_data[:,0])), int(max(Traffic_data[:,0]))):
        DFindxtmp = DFindx
        rows = torch.empty(0, NFeatures, device=device) # Nnodes and NFeatures are predefined fixed values
        while Traffic_data[DFindx,0] == fr:
            DFindx += 1

        rows = Traffic_data[DFindxtmp:DFindx, Columns_to_keep] # We only keep the columns that we need
        Fr.append(fr)  # We keep the track of the frame number
        IDs.append(Traffic_data[DFindxtmp:DFindx, 1])
        Zones.append(Traffic_data[DFindxtmp:DFindx, 11])
        while rows.size()[0] < Nusers:  # This is to make sure that rows is [Nnodes, NFeatures]
            rows = torch.cat((rows, torch.zeros(1, NFeatures, device=device)), dim=0)

        Scene.append(rows)
        if len(Scene) % scenelet_len == 0: # This only happens after each scenelet_len frames
            trORtst = checktstvstr(fr, indices, scenelet_len, train_size, test_size) # Find if the Scene belongs to train, test, or validation
            Scene = torch.stack(Scene, dim=0).to(device=device)
            Scene, Zones, IDs, NObjs = ConsistensyCheck(Scene, Zones, IDs, Nnodes, NFeatures) # Now we sort the agents to be on the same row in the whole Scene
            if trORtst == 1: # test
                Scenetst.Slide_(Scene, Zones, IDs, Fr, NObjs, dssc)
            if not only_test: # 
                if trORtst == 0:
                    Scenetr.Slide_(Scene, Zones, IDs, Fr, NObjs)
                elif trORtst == 2:
                    Sceneval.Slide_(Scene, Zones, IDs, Fr, NObjs)
            Scene = []
            Fr = []
            IDs = []
            Zones = []
    torch.save(indices, os.path.join(os.getcwd(),'Pickled',f'indices.pt'))
    return Scenetr, Scenetst, Sceneval

def ConsistensyCheck(Scene, Zones, IDs, Nnodes, NFeatures): # Makes sure that the rows related to the same ID are in the same order in the  whole lists
    globlist = torch.cat(IDs, dim=0).unique() # Base list to compare the rest of the lists
    len_globlist = max(len(globlist), Nnodes)
    device = Scene.device
    SortedScene = []
    SortedZones = []
    SortedIDs = []

    for k, ids in enumerate(IDs): # IDs are spread over the frames, we need to sort them frame by frame
        Sc = torch.zeros(len_globlist, NFeatures, device=device) # tmp Scene which will be sorted
        Zonelist = 200*torch.ones(len_globlist, device=device) # Zone 200 is the default zone that meant to be ignored
        _, indices = torch.where(ids.unsqueeze(1)==globlist) 
        Sc[indices] = Scene[k, :len(indices)] # Swap the rows based on the indices
        Zonelist[indices] = Zones[k] # Swap the zones based on the indices
        SortedZones.append(Zonelist[:Nnodes]) # Sort the Zones based on the indxlist
        SortedScene.append(Sc[:Nnodes]) # We dump the rest of the agents that are over Nnodes
    SortedIDs = globlist.repeat(k, 1) # Sort the IDs based on the indxlist
    SortedScene = torch.stack(SortedScene, dim= 0).to(device=device)
    SortedZones = torch.stack(SortedZones, dim=0).to(device=device)
    return SortedScene, SortedZones, SortedIDs, len(globlist)



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

def DataLoader_Scene(datasettr, datasettst, datasetval, batch_size):
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

def Zoneconf(path = '/utilz/ZoneConf.yaml'):
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")