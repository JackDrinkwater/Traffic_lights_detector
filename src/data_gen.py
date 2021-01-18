import numpy as np 
import cv2 as cv
import pandas as pd
import random
from math import ceil
from sklearn.utils import shuffle

dt = np.dtype(np.float32)
def generator(batch_size=50):
    while True:
        for j in range(batch_size):
            Xs = []
            Ys = []
            count = 0
            
            while count < 100:
                
                day_or_night = random.randint(0,1)

                if day_or_night == 0: 
                    folder_day = random.randint(1,13)
                    path_0 = '/Users/stanislav/Downloads/archive/Annotations/Annotations/dayTrain/dayClip{}/frameAnnotationsBOX.csv'.format(folder_day)
                    csv_file = pd.read_csv(filepath_or_buffer=path_0, sep=';')

                else:
                    folder_night = random.randint(1,5)
                    path_0 = '/Users/stanislav/Downloads/archive/Annotations/Annotations/nightTrain/nightClip{}/frameAnnotationsBOX.csv'.format(folder_night)
                    csv_file = pd.read_csv(filepath_or_buffer=path_0, sep=';')

                # choose picture 
                i = random.randint(0, len(csv_file.iloc[:,0].unique())-1)# choose random number of picture in folder
                full_pic_name = csv_file.iloc[:,0].unique()[i] # with index above choose full name picture 
                pic_name = csv_file.iloc[:,0].unique()[i].split('/')[1] # with index above choose picture 

                if day_or_night == 0:
                    path_to_img = '/Users/stanislav/Downloads/archive/dayTrain/dayTrain/dayClip{}/frames/'.format(folder_day) + pic_name
                else:
                    path_to_img = '/Users/stanislav/Downloads/archive/nightTrain/nightTrain/nightClip{}/frames/'.format(folder_night) + pic_name

                img = cv.imread(path_to_img)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                # find coordinates
                number_of_same_pic = len(csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[:,0]) # how many pic with same name
                img = cv.copyMakeBorder(img, 200, 200, 200, 200, cv.BORDER_REPLICATE) 
                
                # blobbing
                params = cv.SimpleBlobDetector_Params()
                params.minThreshold = 1
                params.maxThreshold = 255
                params.filterByArea = True
                params.minArea = 100 # for day it was 300
                #params.maxArea = 1500
                params.filterByCircularity = False
                params.filterByConvexity = False
                params.filterByInertia = False

                detector = cv.SimpleBlobDetector_create(params)
                keypoints = detector.detect(img)

                kps = np.array([key for key in keypoints])
                #print(kps)

                #im_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                

                for i in range(number_of_same_pic):
                    if count < 100:
                        #appear = True
                        # coors of box
                        x1 = csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[i,2]+200
                        y1 = csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[i,3]+200
                        x2 = csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[i,4]+200
                        y2 = csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[i,5]+200
                        
                        # condition for keypoins which are not boxes - TAKES MUCH TIME
                        for key in keypoints:
                            keypoints = [key for key in keypoints if not ((x1-50 < key.pt[0] < x2+50) and (y1-50 < key.pt[1] < y2+50))]     
                        
                        random_crop_x1 = random.randint(0, 200-(x2-x1))
                        random_crop_x2 = 200 - random_crop_x1
                        random_crop_y1 = random.randint(0, 200-(y2-y1))
                        random_crop_y2 = 200 - random_crop_y1

                        cropped_img = img[y1-random_crop_y1:y2+random_crop_y2, x1-random_crop_x1:x2+random_crop_x2]
                        
                        new_x1 = random_crop_x1
                        new_y1 = random_crop_y1
                        new_x2 = new_x1 + (x2-x1)
                        new_y2 = new_y1 + (y2-y1)

                        w = cropped_img.shape[1]
                        h = cropped_img.shape[0]

                        Rx = (64 / w)
                        Ry = (64 / h)
                        
                        x1 = ceil(new_x1*Rx)
                        y1 = ceil(new_y1*Ry) 
                        x2 = ceil(new_x2*Rx)
                        y2 = ceil(new_y2*Ry)

                        cropped_img = cv.resize(cropped_img, (64, 64))
                        cropped_img = cropped_img.reshape(1, 64, 64, 3)
                        box = np.array([1, x1, y1, x2, y2], dtype=dt)
                        
                        #image_rect = cv.rectangle(cropped_img, (x1,y1), (x2,y2), (0,255,0), 1)
                        
                        Xs.append(np.array(cropped_img, dtype=dt) / 255.), Ys.append(box)
                        #box = box.reshape(5, 1)
                        
                        count += 1
                                                        
                        #plt.imshow(image_rect)
                        #plt.show()
                keypoints = keypoints[-5:-1]
                for k in range(len(keypoints)):
                    if count < 100:
                        #appear = False
                        k_x1 = int(round(keypoints[k].pt[0]-100))
                        k_y1 = int(round(keypoints[k].pt[1]-100))
                        k_x2 = int(round(keypoints[k].pt[0]+100))
                        k_y2 = int(round(keypoints[k].pt[1]+100))

                        cropped_img = img[k_y1:k_y2, k_x1:k_x2]
                        cropped_img = cv.resize(cropped_img, (64, 64))
                        cropped_img = cropped_img.reshape(1, 64, 64, 3)

                        box = np.array([0, 0, 0, 0, 0], dtype=dt)
                        #box = box.reshape(5, 1)
                                                        
                        #plt.imshow(cropped_img)
                        #plt.show()

                        Xs.append(np.array(cropped_img, dtype=dt) / 255.), Ys.append(box)

                        count += 1
                        
            Xs, Ys = shuffle(Xs, Ys)            
            yield Xs, Ys
