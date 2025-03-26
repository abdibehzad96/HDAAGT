** Make sure you use the same 'ZoneConf.yaml' file, we store it in "utilz/ZoneConf.yaml"


#1 To start from scratch, We Start with the Modified detection data:
        Objectpath = '/home/abdikhab/All Data/Stone Church/ModifiedData.csv'
        framepath = '/home/abdikhab/All Data/Stone Church/Traffic Light StoneChurch.csv'
    Then using "utilz/trafficlight.py" we create:
        finalpath = '/home/abdikhab/All Data/Stone Church/Trafficlightadded.csv'


#2 If we want a calibration data: we run the "~/calibration/pixel2cord.ipynb"
    Objectpath = '/home/abdikhab/Calibration/Trafficlightadded.csv'
    Finalpath = '/home/abdikhab/Calibration/TrfZonXYCam.csv'
#2 Second we run the main.py
    filepath = '/home/abdikhab/Calibration/TrfZonXYCam.csv'