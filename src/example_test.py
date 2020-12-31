import numpy as np
import cv2 as cv

path_to_image = 'data/example.jpg'
path_to_model = 'models/model.h5'
image = cv.imread(path_to_image)
model = load_model(path_to_model)

def predict_example(model, image): 
	""" Generate exmaple picture and predict class and boundary box """
	img = cv.copyMakeBorder(image, 40, 40, 40, 40, cv.BORDER_REPLICATE)

    # blobbing parameters
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

    for z in range(len(keypoints)):
        new_x1 = int(round(keypoints[z].pt[0])-32) 
        new_y1 = int(round(keypoints[z].pt[1])-32)
        new_x2 = int(round(keypoints[z].pt[0])+32)
        new_y2 = int(round(keypoints[z].pt[1])+32)

        cropped_im = img[new_y1:new_y2, new_x1:new_x2]
        cropped_im = np.expand_dims(cropped_im, 0) / 255.
        predictions = model.predict(cropped_im)

        if predictions[0][0] > 0.5:
        	img = cv.rectangle(img, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 1) 
        	
    plt.imshow(img)
    plt.show()	

    return img


