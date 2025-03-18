
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
#create a class ID with init function to initialize the variables

class Objects:
    def __init__(self, ID,Fr, BB,TrL):        
        self.ID = ID
        self.Fr = Fr
        self.BB = BB
        self.TrL = TrL
        self.neighbours = [] # list of all neighbours with their ID, BBx and BBy
        self.zone = self.zonefinder(BB, ZoneConf) # zone of the object
        self.canditatesN = [] # canditates for neighbours

    def addNeighbours(self,R, NID, NBB):
        if R:
            self.neighbours.append([R, NID, NBB])
    def rankNeighbours(self): 
        self.neighbours.sort(key= lambda x: x[0]) # sort the neighbours based on the distance
        while len(self.neighbours) < 5:
            self.neighbours.append([0,0,[0,0]])
    def zonefinder(self, BB, Zones):
        for i, zone in enumerate(Zones):
            Poly = Polygon(zone)
            if Poly.contains(Point(BB[0], BB[1])):
                return i
        return 100

class RedFlag:
    def __init__(self, ID, Fr, BBx,BBy ,n):
        self.ID = ID
        self.Fr = Fr
        self.BBx = BBx
        self.BBy = BBy
        self.n = n

def movingavg(x, y):
    order = 5  # Filter order
    b = np.full(order, 1.0/order)
    x =np.floor(np.convolve(x, b, mode='same'))
    y =np.floor(np.convolve(y, b, mode='same'))

    return x, y

class Vehicles_in_Frame:
    def __init__(self):
        self.RedFlag = []
        self.vehicles = []
        self.tracks = {} # hold the index of the vehicles in the self.vehicles which have same ID
        self.moved = {}
        self.frames = {}    # hold the index of the vehicles in the self.vehicles which are in the same frame
        self.l = 0
    def CheckExist(self, ID, Fr = None):
        if Fr:
            for obj in self.vehicles:
                if obj.ID == ID and obj.Fr == Fr:
                    return True
            return False
        else:
            for obj in self.vehicles:
                if obj.ID == ID:
                    return True
            return False
    
    def checkMoved(self, ID):
        movement = np.zeros((5,1))
        track = self.tracks[ID]
        j = 0
        for n, inx in enumerate(track[1:]):
            if self.vehicles[inx].zone != self.vehicles[track[n]].zone:
                movement[self.vehicles[inx].zone] = 1
                movement[self.vehicles[track[n]].zone] = 1
                j += 1
        if j > 2:
            bbx = []
            bby = []
            ff = []
            #print("Red Flag on ID: ", ID, " at frame: ", self.vehicles[track[n]].Fr, "with movement of: ", n)
            for i in self.tracks[ID]:
                bbx.append(self.vehicles[i].BB[0])
                bby.append(self.vehicles[i].BB[1])
                ff.append(self.vehicles[i].Fr)
            self.RedFlag.append(RedFlag(ID, ff, bbx,bby, n))
            print("Red Flag on ID: ", ID, " at frame: ", self.vehicles[track[n]].Fr, "with movement of: ", n)
        else :
            if np.sum(movement) > 2:
                self.moved[ID] = np.sum(movement)
                return True
            else:
                self.moved[ID] = np.sum(movement)
        return False

    def addVehicle(self, ID, Fr, BB, TrL):
        NewObj = Objects(ID, Fr, BB, TrL)
        if NewObj.zone != 100:
            if ID in self.tracks:      #check if the self.tracks[ID] exists or not
                self.tracks[ID].append(self.l)
            else:
                self.tracks[ID] = [self.l]
            
            if Fr in self.frames:   #check if the self.frames[Fr] exists or not
                self.frames[Fr].append(self.l)
            else:
                self.frames[Fr] = [self.l]
            self.vehicles.append(NewObj)
            self.l += 1

    def eligibility(self, trgt,nghbr, criteria = 300):
        R=math.sqrt((trgt[0] -nghbr[0] )**2 + (trgt[1] - nghbr[1])**2)
        if R < criteria:
            return int(R)
        return False

    def checkNeighbours(self): # only needs to be called once                                                                           
        for obj in self.vehicles:
            for ids in self.vehicles:
                if obj.ID != ids.ID and obj.Fr == ids.Fr:
                    R = self.eligibility(obj.BB, ids.BB)
                    obj.addNeighbours(R, ids.ID, ids.BB)
            obj.rankNeighbours()
    
    def checkNeighboursFr(self, Fr): # can be called per each Fr                                                                     
        for obj in self.frames[Fr]:
            for sub in self.frames[Fr]:
                if self.vehicles[obj].ID != self.vehicles[sub].ID:
                    R = self.eligibility(self.vehicles[obj].BB, self.vehicles[sub].BB)
                    self.vehicles[obj].addNeighbours(R, self.vehicles[sub].ID, self.vehicles[sub].BB)
            self.vehicles[obj].rankNeighbours()
    
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

    
    def nextCandidate(self, Inx, NC = 5):
        self.vehicles[Inx[0]].canditatesN = self.vehicles[Inx[0]].neighbours[:NC]
        j = 0
        for f in Inx[1:]:
            j += 1
            available = list(range(NC))
            #if Fr[j+1] == Fr[j]+1:       # if the frame number is the next frame
            # while len(self.vehicles[f].neighbours) <= NC:
            #     self.vehicles[f].addNeighbours(3000,0,[0,0])
            # 
            tmp = self.vehicles[f].neighbours
            prevN= self.vehicles[Inx[j-1]].canditatesN
            for i in range(NC):
                for n in range(NC):
                #swap values of the tuple
                    if prevN[i][0] == tmp[n][0] and prevN[i][0]!=0:
                        #tmp.insert(i,tmp.pop(n))
                        available[n] = None
                        #reserved.append(n)
                        #assert self.vehicles[f].canditatesN == None, f"len of {f}vehicle {self.vehicles[f].ID} is {self.vehicles[f].canditatesN}"
                        self.vehicles[f].canditatesN.insert(n, tmp[n])
                        break
            for i in available:
                if i != None:
                    self.vehicles[f].canditatesN.insert(i, tmp[i])

    def saveAll(self, ct, filename ='P-'):
        
        filename = filename + ct + '.pkl'
        path = os.path.join(os.getcwd(),'Pickled', filename)
        # Save the DataFrame to a file using pickle
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def csvLog(self,ct, cwd = os.getcwd()):
        # n = []
        # for files in os.listdir(cwd):
        #     if files.endswith('.csv') and files.startswith('BehzadProcessed'):
        #         index = files.find('.csv')
        #         #check if the index=15 is a char or not
        #         if index > 15:
        #             n.append(int(files[15:index]))
        #         else:
        #             n.append(0)
        # if n == []:
        #     m = 0
        # else:
        #     m = max(n) + 1
        #Pros_path = os.path.join(cwd, f'BehzadProcessed{m}.csv')
        Pros_path = os.path.join(cwd, 'Processed',"Zones" + ct + ".csv" )
        with open(Pros_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for Ids in self.tracks:
                rBBx, rBBy , rTr1, rTr2, rTr3, rTr4, N1x, N1y, N2x, N2y, N3x, N3y, N4x, N4y, N5x, N5y = [], [], [], [],[], [], [], [],[], [], [], [],[], [], [], [] 
                if self.checkMoved(Ids) and len(self.tracks[Ids]) > 20:
                    self.nextCandidate(self.tracks[Ids])
                    for v in self.tracks[Ids]:
                        rBBx.append(self.vehicles[v].BB[0])
                        rBBy.append(self.vehicles[v].BB[1])
                        rTr1.append(self.vehicles[v].TrL[0])
                        rTr2.append(self.vehicles[v].TrL[1])
                        rTr3.append(self.vehicles[v].TrL[2])
                        rTr4.append(self.vehicles[v].TrL[3])
                        N1x.append(self.vehicles[v].canditatesN[0][2][0])
                        N1y.append(self.vehicles[v].canditatesN[0][2][1])
                        N2x.append(self.vehicles[v].canditatesN[1][2][0])
                        N2y.append(self.vehicles[v].canditatesN[1][2][1])
                        N3x.append(self.vehicles[v].canditatesN[2][2][0])
                        N3y.append(self.vehicles[v].canditatesN[2][2][1])
                        N4x.append(self.vehicles[v].canditatesN[3][2][0])
                        N4y.append(self.vehicles[v].canditatesN[3][2][1])
                        N5x.append(self.vehicles[v].canditatesN[4][2][0])
                        N5y.append(self.vehicles[v].canditatesN[4][2][1])
                        
                    rBBx, rBBy = movingavg(rBBx, rBBy)
                    writer.writerow(rBBx)
                    writer.writerow(rBBy)
                    writer.writerow(rTr1)
                    writer.writerow(rTr2)
                    writer.writerow(rTr3)
                    writer.writerow(rTr4)
                    writer.writerow(N1x)
                    writer.writerow(N1y) 
                    writer.writerow(N2x)
                    writer.writerow(N2y)
                    writer.writerow(N3x)
                    writer.writerow(N3y)
                    writer.writerow(N4x)
                    writer.writerow(N4y)
                    writer.writerow(N5x)
                    writer.writerow(N5y)
                    writer.writerow([self.vehicles[self.tracks[Ids][-1]].zone])
        
        with open(os.path.join(cwd, 'Processed',"RedFlag" + ct + ".csv" ), 'w', newline='') as file:
            writer = csv.writer(file)
            for obj in self.RedFlag:
                writer.writerow([obj.n, obj.ID])
                writer.writerow(obj.BBx)
                writer.writerow(obj.BBy)
                writer.writerow(obj.Fr)
    



    @staticmethod
    def load_DF(filename):
        # Load the DataFrame from the file using pickle
        with open(filename, 'rb') as file:
            return pickle.load(file)
    



# load .yaml file containing 4 points of the 4 zones
ZoneConf = []
with open('ZoneConf.yaml') as file:
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
LoadSaved = True
FromScratch = not LoadSaved

Objects_in_Frame = Vehicles_in_Frame()
if FromScratch:
    cwd = os.getcwd()
    file_path = os.path.join(cwd,'RawData','Detection_Arsalan_Labled.csv')
    #laod the csv file into a dataframe
    df = pd.read_csv(file_path, dtype =float)
    for n in range(int(min(df['Frame'])), int(max(df["Frame"]))):
        a = df[df['Frame'] == n].reset_index() # List of objects in frame n
        # if n > 50:
        #      break
        for i in range(len(a)):
            Objects_in_Frame.addVehicle(a["ID"][i], a["Frame"][i],[a["BBx"][i], a["BBy"][i]], [a["Tr1"][i],a["Tr2"][i],a["Tr3"][i],a["Tr4"][i]])
        
        Objects_in_Frame.checkNeighboursFr(n)
        #print("All Objects are added to the class")
        #Objects_in_Frame.checkNeighbours()
        #print("All neighbours are added to the class")
        Objects_in_Frame.checkTrafficLightFr(n)
    print("All traffic lights are checked")
    print("Saving the data")
    ct = datetime.datetime.now().strftime(r"%Y%m%dT%H%M")
    Objects_in_Frame.csvLog(ct)
    print("Data is saved at: ", ct)
    # print("Saving the class")
    # Objects_in_Frame.saveAll(ct)

if LoadSaved:
    print("Loading the class")
    Objects_in_Frame = Vehicles_in_Frame.load_DF('Pickled/P-20231221T2134.pkl')
    print("All Objects are added to the class")
    print("Checking the Traffic Lights")
    Objects_in_Frame.checkTrafficLight()
    ct = datetime.datetime.now().strftime(r"%Y%m%dT%H%M")
    print("Data is saved at: ", ct)
    Objects_in_Frame.csvLog(ct)
    print("Saving the class")
    # Objects_in_Frame.saveAll()
    # print("Class is saved at: ", ct)

