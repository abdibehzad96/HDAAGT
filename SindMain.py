from utilz.DataReader import *
import os
import torch
from torch.utils.data import Dataset, DataLoader

class Scenes(Dataset):
    def __init__(self, sl,future, Nnodes, input_size, device): # Features include ['BBX','BBY','W', 'L' , 'Cls', 'Xreal', 'Yreal']
        self.Scene = torch.empty(0, sl, Nnodes, input_size, device=device) #[Sampeles, Sequence Length, Nnodes (30), NFeatures(tba)]
        self.Adj_Mat = torch.empty(0, sl, Nnodes, Nnodes, device=device)
        self.Target = torch.empty(0, future, Nnodes, input_size, device=device) #[Sampeles, Sequence Length, Nnodes (30), NFeatures(tba)]

    def add(self, scene, target, adj_mat):
        self.Scene = torch.cat((self.Scene, scene.unsqueeze(0)), dim=0)
        self.Target = torch.cat((self.Target, target.unsqueeze(0)), dim=0)
        self.Adj_Mat = torch.cat((self.Adj_Mat, adj_mat.unsqueeze(0)), dim=0)

    def save(self, path):
        torch.save(self.Scene, os.path.join(path, 'Scene.pt'))
        torch.save(self.Target, os.path.join(path, 'Target.pt'))
        torch.save(self.Adj_Mat, os.path.join(path, 'Adj_Mat.pt'))
    
    def load(self, path):
        self.Scene = torch.load(os.path.join(path, 'Scene.pt'))
        self.Target = torch.load(os.path.join(path, 'Target.pt'))
        self.Adj_Mat = torch.load(os.path.join(path, 'Adj_Mat.pt'))

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

NUsers = 64
NFeatures = 8
sl = 20
future = 30
sl2 = sl//2
tot_len = sl + future
Scene_clss = Scenes(sl,future,NUsers,NFeatures,device)
scene = torch.zeros(tot_len,NUsers,NFeatures, device=device)
global_list = torch.zeros(NUsers)
# path = '/home/abdikhab/New_Idea_Traj_Pred/sinD/Data/Chongqing/6_22_NR_1'
# lightpath = '/home/abdikhab/New_Idea_Traj_Pred/sinD/Data/Chongqing/6_22_NR_1/TrafficLight_06_22_NR1_add_plight.csv'
path = '/home/abdikhab/New_Idea_Traj_Pred/sinD/Data/Changchun/changchun_pudong_507_009'
lightpath = '/home/abdikhab/New_Idea_Traj_Pred/sinD/Data/Changchun/changchun_pudong_507_009/Traffic_Lights.csv'


Veh_tracks_dict, Ped_tracks_dict = read_tracks_all(path)
max_frame = 0
min_frame = 1000
for _, track in Veh_tracks_dict.items():
    frame= torch.from_numpy(track["frame_id"]).max()
    max_frame = frame if frame > max_frame else max_frame
    min_frame = frame if frame < min_frame else min_frame
print("Max frame",max_frame, "Min frame", min_frame)

_, light = read_light(lightpath, max_frame)
more_list = []
for Frme in range(min_frame,max_frame, 10):
    scene = torch.zeros(tot_len,NUsers,NFeatures, device=device)
    adj_mat = torch.zeros(sl,NUsers,NUsers, device=device)
    global_list = {}
    order = 0
    last_Frame = Frme + sl + future
    more = 0
    for _, track in Veh_tracks_dict.items():
        id = torch.tensor(track["track_id"])
        frame= torch.from_numpy(track["frame_id"])
        x = torch.from_numpy(track["x"])
        y = torch.from_numpy(track["y"])
        vx = torch.from_numpy(track["vx"])
        vy = torch.from_numpy(track["vy"])
        yaw_rad = torch.from_numpy(track["yaw_rad"])
        heading_rad = torch.from_numpy(track["heading_rad"])
        if last_Frame in frame:
            for fr in range(sl2):
                real_frame = fr+Frme
                if real_frame in frame and id not in global_list:
                    indices = (frame >= real_frame) * (frame < last_Frame)
                    st_indx = torch.where(frame == real_frame)
                    end_indx = torch.where(frame == last_Frame)
                    if order < NUsers:
                        global_list[id] = order
                        ll = light[last_Frame-sum(indices):last_Frame]
                        scene[fr: tot_len,order] = torch.stack([x[indices],y[indices],vx[indices],vy[indices],yaw_rad[indices],heading_rad[indices],ll[:,0], ll[:,1]],dim=1)
                        order +=1
                        break
                    else:
                        more += 1
    adj_mat[:,:order, :order] = torch.eye(order, order, device=device).unsqueeze(0).repeat(sl, 1, 1)
    Scene_clss.add(scene[:sl], scene[sl:], adj_mat)
    more_list.append(more)
    


print("done")
final_path = '/home/abdikhab/New_Idea_Traj_Pred/sinD/Data/Changchun/'
