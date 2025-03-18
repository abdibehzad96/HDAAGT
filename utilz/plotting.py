#This file is used to plot the trajectories of the preprocessed data
import matplotlib.pyplot as plt
import os
import math
import cv2
import yaml
import re
import numpy as np


class plotting:
    def __init__(self):
        self.data = []
        self.ZoneConf = []
    def plot_Neighb(self,n):
        # while not pressed ESC key continue plotting
        lst = [[0,1],[6,7],[8,9],[10,11],[12,13],[14,15]]
        for i in lst:
            x = self.data[n][i[0]]
            y = self.data[n][i[1]]
            while 0 in x:
                x.remove(0)
                y.remove(0)
            plt.plot(x,y)
        img = plt.imread("16.jpg")
        plt.imshow(img)
        plt.show()

    def plot_Trj(self, n = 0):
        i =0
        for dx in self.data:
            plt.plot(dx[0],dx[1])
            # add id to the plot
            plt.text(50,30,str(i), fontsize=12, color = 'white')
            i +=1
            img = plt.imread("16.jpg")
            # add the img to plot
            plt.imshow(img)
            plt.pause(0.01)
            plt.clf()


    def Imgcreator(self): # save the plot as a video
        print("Creating images")
        img = cv2.imread("16.jpg")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        i = 0
        for dx in self.data:
            # Add the plot to the video
            i +=1
            plt.imshow(image)
            plt.plot(dx[0],dx[1])
            plt.text(40,30,str(i), fontsize=12, color = 'white')
            filename = os.getcwd() + f"/Img2/frame_{i:03d}.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            # Read the saved image and append to frames list
        print("Images created successfully")

    
    def create_video(self):
        print("Creating video")
        cwd = os.getcwd()
        video_path = "CheckTrajectories.mp4"
        list = os.listdir(os.getcwd() + "/Img2")
        list.sort()
        img = cv2.imread("frame_677.png")
        height, width, _ = img.shape
        video = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'mp4v'),0.5,(width,height))
        for lst in list:
            images = cv2.imread(os.path.join(cwd,"Img2",lst))
            video.write(images)
        video.release()
        print("Video created successfully")


    def load(self):
        df =[]
        cwd = os.getcwd()
        csvpath = os.path.join(cwd,'Processed','Trj20231212T1747.csv')
        with open(csvpath, 'r',newline='') as file:
            for line in file:
                row = line.strip().split(',')
                rowf = [float(element) for element in row]
                rowf = [0 if math.isnan(x) else x for x in rowf]
                df.append(rowf)
        for i in range(len(df)//16): #len(df)//16
            self.data.append(df[i*16:(i+1)*16])
        print("Data loaded successfully")
    def zones(self):
        with open('ZoneConf.yaml') as file:
            ZonesYML = yaml.load(file, Loader=yaml.FullLoader)
            #convert the string values to float
            for _, v in ZonesYML.items():
                lst = []
                for _, p  in v.items():    
                    for x in p[0]:
                        b = re.split(r'[,()]',p[0][x])
                        lst.append([int(b[1]), int(b[2])])
                self.ZoneConf.append(np.array(lst))
        print("Zones loaded successfully")

    def plot_zones(self):
        img = cv2.imread("16.jpg")
        for zone in self.ZoneConf:
            # Draw the polygon on the image
            for point in zone:
                print(point)
                cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
        cv2.imshow("Polygon", img)

if __name__ == '__main__':
    plotter = plotting()
    plotter.load()
    #plotter.zones()
    #plotter.plot_zones()
    plotter.Imgcreator()
    #plotter.plot_Trj()

    plotter.create_video()