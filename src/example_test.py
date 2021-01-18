import numpy as np
import cv2 as cv

path_to_image = 'data/example.jpg'
path_to_model = 'models/model.h5'
image = cv.imread(path_to_image)
model = load_model(path_to_model)

def predict_example(model, image): 
	""" Generate exmaple picture and predict class and boundary box """
	img = cv.copyMakeBorder(image, 40, 40, 40, 40, cv.BORDER_REPLICATE)
    image_with_boxes = img.copy()

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
        # finding ROI
        x1 = int(round(keypoints[z].pt[0])-32) 
        y1 = int(round(keypoints[z].pt[1])-32)
        x2 = int(round(keypoints[z].pt[0])+32)
        y2 = int(round(keypoints[z].pt[1])+32)

        # scale coefficients
        Rx = img.shape[1] / 64. 
        Ry = img.shape[0] / 64.

        cropped_im = img[new_y1:new_y2, new_x1:new_x2]
        cropped_im = np.expand_dims(cropped_im, 0) / 255.
        predictions = model.predict(cropped_im)

        if predictions[0][0] > 0.5:
        	image_with_boxes = cv.rectangle(image_with_boxes, (pred[1]*Rx, pred[2]*Ry), (pred[3]*Rx, pred[4]*Ry), (0, 255, 0), 1) 
    
    plt.imshow(image_with_boxes)
    plt.show()	

    return image_with_boxes


