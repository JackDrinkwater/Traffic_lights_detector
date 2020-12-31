import cv2 as cv
import pandas as pd

def train_gen():
    """ Generate train data - make sampling and augmentation then yield X, Y """
    for j in range(1):  # for example, 1 batch
        count = 0
        while count < 100:  # batch-size
            image_path= '/Users/stanislav/Downloads/archive/Annotations/Annotations/dayTrain/dayClip5/frameAnnotationsBOX.csv'
            csv_file = pd.read_csv(filepath_or_buffer=image_path, sep=';')

            # choose random picture 
            i = random.randint(0, len(csv_file.iloc[:,0].unique()))# choose random number of picture in folder
            full_pic_name = csv_file.iloc[:,0].unique()[i] # with index above choose full name picture 
            pic_name = csv_file.iloc[:,0].unique()[i].split('/')[1] # with index above choose picture 
            path_to_img = '/Users/stanislav/Downloads/archive/dayTrain/dayTrain/dayClip5/frames/' + pic_name
            img = cv.imread(path_to_img)

            # how many boxes on one picture
            number_of_same_pic = len(csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[:,0]) 
            for i in range(number_of_same_pic): 
                x1 = csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[i,2]
                y1 = csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[i,3]
                x2 = csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[i,4]
                y2 = csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[i,5]
                img = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # it makes picture usable if traffic light is near the boundary of picture
            img = cv.copyMakeBorder(img, 40, 40, 40, 40, cv.BORDER_REPLICATE)

            params = cv.SimpleBlobDetector_Params()
            params.minThreshold = 1
            params.maxThreshold = 255
            params.filterByArea = True
            params.minArea = 100 
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = False

            detector = cv.SimpleBlobDetector_create(params)
            keypoints = detector.detect(img)

            # crop samples based on keypoints of blobs
            for z in range(len(keypoints)):
                if count < 100: 
                    new_x1 = int(round(keypoints[z].pt[0])-32) 
                    new_y1 = int(round(keypoints[z].pt[1])-32)
                    new_x2 = int(round(keypoints[z].pt[0])+32)
                    new_y2 = int(round(keypoints[z].pt[1])+32)
                    cropped_im = img[new_y1:new_y2, new_x1:new_x2]
                    # Here will be the code of 1) flipping as augmentation
                    # and 2) making X as samples and Y as coordinates of 
                    # boundary box.
                    # Then yeild X / 255. , Y 
                    plt.imshow(cropped_im)
                    plt.show()
                    count += 1
