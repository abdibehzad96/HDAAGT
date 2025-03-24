import os
import cv2
import csv
import pandas as pd
import numpy as np
def create_video(impath, vidpath):
    print("Creating video")
    list = os.listdir(impath)
    list.sort()
    img = cv2.imread(os.path.join(impath, list[0]))
    height, width, _ = img.shape
    video = cv2.VideoWriter(os.path.join(vidpath, "StoneChurch.mp4"),cv2.VideoWriter_fourcc(*'mp4v'),10,(width,height))
    n = 0
    for lst in list:
        n += 1
        img = cv2.imread(os.path.join(impath, lst))
        cv2.putText(img, f'Frame {n}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA) 
        cv2.imwrite(os.path.join(vidpath, f"{n}.jpg"), img)
        video.write(img)
    video.release()
    print("Video created successfully")


def csvadd(Objectpath, framepath, finalpath):
    headerfinal = ['Frame', 'ID', 'X', 'Y', 'Width', 'Height', 'Class', 'Tr1', 'Tr2', 'Tr3', 'Tr4']
    headerframe = ['Frame', 'Tr1', 'Tr2', 'Tr3', 'Tr4']
    # read csv file using pandas, first row is the header
    obj = pd.read_csv(Objectpath, header=None).to_numpy()
    finalfile = []
    Error = False
    # read csv file of framepath, frame contains the frame number,Tr1, Tr2, Tr3, Tr4 columns. First row is the header
    frame = pd.read_csv(framepath)
    col = 0
    k=0
    frlist =frame['Frame'][1:].to_numpy()
    frame = frame.to_numpy()
    startfr= frame[0][0]
    for fr in frlist:
        # find the row of the frame in the object file
        Trf = frame[k][1:]
        k += 1
        if obj[col][0] == startfr:
            while obj[col][0] <= fr:
                finalfile.append(np.append(obj[col], Trf))
                col += 1
            startfr = fr+1 # the next frame number that must be found to start the next iteration
        else:
            print(f"Frame{startfr} not found")
            Error = True
            break
    if Error == False:
        # write the final file to the csv file
        print("Writing to the traffic light added csv file")
        # check if the file exists, if it does, remove it
        if os.path.exists(finalpath):
            os.system(f"rm {finalpath}")
            print("File removed, Creating new file")
        with open(finalpath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headerfinal)
            for row in finalfile:
                writer.writerow(row)

        print("Writing completed")
    else:
        print("Error occured, check the frame number")
        print(f"Frame number not found: {startfr}")


    
# impath = '/home/abdikhab/All Data/Stone Church/Images'
# vidpath = '/home/abdikhab/All Data/Stone Church/Videos'
# create_video(impath, vidpath)

Objectpath = '/home/abdikhab/All Data/Stone Church/test.csv'
framepath = '/home/abdikhab/All Data/Stone Church/frames.csv'
finalpath = '/home/abdikhab/All Data/Stone Church/Trafficlightadded.csv'
csvadd(Objectpath, framepath, finalpath)