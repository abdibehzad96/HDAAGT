# This code creates the same trajectory data as the revious version, but calculates the speed, zone, ID and frames of the vehicles
# The structure of the code is as follows: Fisheye X,Y, TrfL1-4, BirdEye X,Y, Vxb, Vxy zone, ID, Frame
import math
import datetime
import os
import pandas as pd
import yaml
from shapely.geometry import Point, Polygon
import re
import csv
import pickle
import numpy as np
# Facts:
# 1. Each road object can be represented as a node in a graph, such as traffic light, road sign, vehicle, pedestrian, etc.
# 2. High order structure graphs can be represented as a set of nodes and edges(simplex), where the edges are the connections between the nodes.
# 3. We can formulate the zones as nodes and the edges will be 1 if the vehilce is inside the zone and 0 if it's outside the zone.
# 4. What if in the definition of adjacency matrix, we put a limitation and if the output was not rational, we put some special value!
# 5. we can consider a probaility of intentions for each vehicle as a feature of the node
#Scene Centric Data Preperation
class Objects:
    def __init__(self, ID,Fr, BB, RealXY,TrL, Zone, cls):        
        self.ID = ID        # The real ID of the object from the dataset
        self.Fr = Fr        # The real frame number of the object from the dataset
        self.BB = BB        # The bounding box of the object from the dataset
        self.Cls = cls     # The class of the object from the dataset
        self.Vx = None      # The velocity of the object in x direction
        self.Vy = None      # The velocity of the object in y direction 
        self.Radios = None  # The radios of the object
        self.sinth = None   # The sin of the object
        self.costh = None   # The cos of the object
        self.TrL = TrL      # The traffic light of the object from the dataset
        self.RealXY = RealXY      # The Real XY of the object from the dataset
        self.Zone = Zone.astype(int)    # The zone of the object from the dataset


class RedFlag:
    def __init__(self, ID, Fr, BBx,BBy ,n):
        self.ID = ID
        self.Fr = Fr
        self.BBx = BBx
        self.BBy = BBy
        self.n = n
        self.direction = None
        self.state = None # 0: stopped, 1: moving, 2: braking

class Scene:
    def __init__(self):
        # self.RedFlag = []
        self.frames = {}    # hold the index of the vehicles in the self.vehicles which are in the same frame
        self.vehicles = []
        self.tracks = {} # hold the index of the vehicles in the self.vehicles which have same ID
        self.moved = {}
        self.l = 0


    def CheckExist(self, ID, Fr = None):
        if Fr: # Checks if the ID exists in the self.vehicles and the frame number is the same
            for obj in self.vehicles:
                if obj.ID == ID and obj.Fr == Fr:
                    return True
            return False
        else: # Check if the ID exists in the self.vehicles
            for obj in self.vehicles:
                if obj.ID == ID:
                    return True
            return False
        
    def direction(self, ID): # trial version, hasn't been thoroughly thought
        track = self.tracks[ID]
        for n, inx in enumerate(track[1:]):
            self.vehicles[inx].direction = self.vehicles[inx].RealXY - self.vehicles[n].RealXY


    def checkMoved(self, ID, ct):
        movement = np.zeros((9,1)) # We defined 13 zones, including the intersection centre and lanes
        track = self.tracks[ID]
        j = 0
        for n, inx in enumerate(track[1:]):
            if self.vehicles[inx].Zone != self.vehicles[track[n]].Zone:
                movement[self.vehicles[inx].Zone] = 1
                movement[self.vehicles[track[n]].Zone] = 1
                j += 1 # movement counter
        if j > 5:
            # bbx = []
            # bby = []
            # ff = []
            #print("Red Flag on ID: ", ID, " at frame: ", self.vehicles[track[n]].Fr, "with movement of: ", n)
            # for i in self.tracks[ID]:
                # bbx.append(self.vehicles[i].BB[0])
                # bby.append(self.vehicles[i].BB[1])
                # ff.append(self.vehicles[i].Fr)
            # self.RedFlag.append(RedFlag(ID, ff, bbx,bby, n))
            savelog(f"Red Flag on ID:  {ID}, at frame: {self.vehicles[track[n]].Fr}, with movement of:  {n}", ct)
        else :
            if np.sum(movement) > 2:
                self.moved[ID] = np.sum(movement)
                return True
            else:
                self.moved[ID] = np.sum(movement)
        return False

    def addVehicle(self, ID, Fr, BB, RealXY, TrL, Zone, Cls):
        NewObj = Objects(ID, Fr, BB, RealXY, TrL, Zone, Cls)
        if NewObj.Zone != 100:
            if ID in self.tracks:      #check if the self.tracks[ID] exists or not, if yes append or create a new one
                self.tracks[ID].append(self.l)
            else:
                self.tracks[ID] = [self.l]
            
            if Fr in self.frames:   #check if the self.frames[Fr] exists or not
                self.frames[Fr].append(self.l)
            else:
                self.frames[Fr] = [self.l]
            self.vehicles.append(NewObj)
            self.l += 1 # l is the lth object in the list

    def checkTrafficLight(self): # only needs to be called once
        for obj in self.vehicles:
            if obj.TrL is None:
                for val in self.vehicles:
                    if val.Fr == obj.Fr -1 and val.Tr:
                        obj.TrL = val.TrL
                        break


    def checkTrafficLightFr(self, Fr): # only needs to be called once
        for obj in self.frames[Fr]:
            if math.isnan(self.vehicles[obj].TrL[0]):
                self.vehicles[obj].TrL = self.vehicles[self.frames[Fr-1][0]].TrL


    def saveAll(self, ct, filename ='P-'):
        filename = filename + ct + '.pkl'
        path = os.path.join(os.getcwd(),'Pickled', filename)
        # Save the DataFrame to a file using pickle
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    
    def movingavg(self, X, y):
        x = np.array(X)
        y = np.array(y)
        x = np.convolve(x, np.ones((3,))/3, mode='valid')
        y = np.convolve(y, np.ones((3,))/3, mode='valid')
        return x, y
    
    def speedcalc(self, rX, rY):
        rX = np.array(rX).append(rX[-1])
        rY = np.array(rY).append(rY[-1])
        rVx = np.diff(rX)
        rVy = np.diff(rY)
        return rVx, rVy
    
    def csvLog(self,ct, cwd = os.getcwd()):
        Pros_path = os.path.join(cwd, 'Processed',"Trj" + ct + ".csv" )
        maxlen = 0
        minlen = 1000
        avglen = 0
        cnt = 0
        with open(Pros_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for Ids in self.tracks:
                rBBx, rBBy , rTr1, rTr2, rTr3, rTr4, rZone, rXbirdeye, rYbirdeye, rID, rFr, rVxF, rVyF = [], [], [], [],[], [], [], [],[], [], [], [], []
                if self.checkMoved(Ids, ct) and len(self.tracks[Ids]) > 20:
                    # self.nextCandidate(self.tracks[Ids])
                    for v in self.tracks[Ids]:
                        rBBx += [self.vehicles[v].BB[0]]
                        rBBy += [self.vehicles[v].BB[1]]
                        rTr1 +=[self.vehicles[v].TrL[0]]
                        rTr2 +=[self.vehicles[v].TrL[1]]
                        rTr3 +=[self.vehicles[v].TrL[2]]
                        rTr4 +=[self.vehicles[v].TrL[3]]
                        rZone += [self.vehicles[v].Zone]
                        rXbirdeye += [self.vehicles[v].RealXY[0]]
                        rYbirdeye += [self.vehicles[v].RealXY[1]]
                        rID += [self.vehicles[v].ID]
                        rFr += [self.vehicles[v].Fr]
                    maxlen = max(maxlen, len(rBBx))
                    minlen = min(minlen, len(rBBx))
                    avglen += len(rBBx)
                    cnt += 1
                    rBBx, rBBy = self.movingavg(rBBx, rBBy)
                    rXbirdeye, rYbirdeye = self.movingavg(rXbirdeye, rYbirdeye)
                    # rVxF, rVyF = self.speedcalc(rXbirdeye, rYbirdeye)
                    # rVx, rVy = self.speedcalc(rXbirdeye, rYbirdeye)
                    writer.writerow(rBBx)
                    writer.writerow(rBBy)
                    writer.writerow(rTr1)
                    writer.writerow(rTr2)
                    writer.writerow(rTr3)
                    writer.writerow(rTr4)
                    writer.writerow(rXbirdeye)
                    writer.writerow(rYbirdeye)
                    # writer.writerow(rVx)
                    # writer.writerow(rVy)
                    writer.writerow(rZone)
                    writer.writerow(rID)
                    writer.writerow(rFr)
        savelog(f"Max: {maxlen}, Min: {minlen}, Avg: {avglen/cnt} length of the trajectories over {cnt}, objects",ct)

        
        # with open(os.path.join(cwd, 'Processed',"RedFlag" + ct + ".csv" ), 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     for obj in self.RedFlag:
        #         writer.writerow([obj.n, obj.ID])
        #         writer.writerow(obj.BBx)
        #         writer.writerow(obj.BBy)
        #         writer.writerow(obj.Fr)
    

def savelog(log, ct): # append the log to the existing log file while keeping the old logs
    # if the log file does not exist, create one
    print(log)
    if not os.path.exists(os.path.join(os.getcwd(),'logs')):
        os.mkdir(os.path.join(os.getcwd(),'logs'))
    with open(os.path.join(os.getcwd(),'logs', f'preprocesslog-{ct}.txt'), 'a') as file:
        file.write('\n' + log)
        file.close()

    @staticmethod
    def load_DF(filename):
        # Load the DataFrame from the file using pickle
        with open(filename, 'rb') as file:
            return pickle.load(file)


# load .yaml file containing the points of the zones
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


# loop through the dataframes which have the same frame number and add them to the class
LoadSaved = False
FromScratch = not LoadSaved
Objects_in_Frame = Scene()
if FromScratch:
    file_path = '/home/abdikhab/New_Idea_Traj_Pred/RawData/TrfZonXYCam.csv'
    ct = datetime.datetime.now().strftime(r"%Y%m%dT%H%M")
    #laod the csv file into a dataframe
    Headers = ['Frame', 'ID', 'BBx', 'BBy','W', 'L' , 'Cls','Tr1', 'Tr2', 'Tr3', 'Tr4', 'Zone', 'Xreal', 'Yreal']
    df = pd.read_csv(file_path, dtype =float)
    df.columns = Headers
    maxobj = 0
    minobj = 1000
    avgobj = 0
    cnt = 0
    for n in range(int(min(df['Frame'])), 1+ int(max(df['Frame']))):
        a = df[df['Frame'] == n].reset_index() # List of objects in frame n
        maxobj = max(maxobj, len(a))
        minobj = min(minobj, len(a))
        avgobj += len(a)
        cnt += 1
        for i in range(len(a)):
            Objects_in_Frame.addVehicle(a['ID'][i], a['Frame'][i],[a['BBx'][i], a['BBy'][i]],[a['Xreal'][i], a['Yreal'][i]], [a['Tr1'][i],a['Tr2'][i],a['Tr3'][i],a['Tr4'][i]], a['Zone'][i], a['Cls'][i])
        Objects_in_Frame.checkTrafficLightFr(n)
    savelog("All traffic lights are checked", ct)
    savelog(f"Saving the data with the following statistics: Max:  {maxobj}, Min:  {minobj}, Avg:  {avgobj/cnt},  number of objects over {cnt}, frames", ct)
    
    Objects_in_Frame.csvLog(ct)
    savelog(f"Data is saved at: {ct}", ct)
    savelog("Saving the class", ct)
    Objects_in_Frame.saveAll(ct)

if LoadSaved:
    print("Loading the class")
    Objects_in_Frame = Scene.load_DF('Vehicles.pkl')
    print("All Objects are added to the class")
    print("Checking the Traffic Lights")
    Objects_in_Frame.checkTrafficLight()
    print("Data is saved at: ", ct)
    Objects_in_Frame.csvLog()
    print("Saving the class")
    Objects_in_Frame.saveAll()
    print("Class is saved at: ", ct)

