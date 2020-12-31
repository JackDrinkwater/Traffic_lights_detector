def real_gen_2():
    for j in range(1):
        count = 0
        while count < 64:
            path_0 = '/Users/stanislav/Downloads/archive/Annotations/Annotations/dayTrain/dayClip5/frameAnnotationsBOX.csv'
            csv_file = pd.read_csv(filepath_or_buffer=path_0, sep=';')
    
    # choose picture 
            i = random.randint(0, len(csv_file.iloc[:,0].unique()))# choose random number of picture in folder
            full_pic_name = csv_file.iloc[:,0].unique()[i] # with index above choose full name picture 
            pic_name = csv_file.iloc[:,0].unique()[i].split('/')[1] # with index above choose picture 
            path_to_img = '/Users/stanislav/Downloads/archive/dayTrain/dayTrain/dayClip5/frames/' + pic_name
    
        # print picture
            img = cv.imread(path_to_img)
            
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


        # find coordinates
            number_of_same_pic = len(csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[:,0]) # how many pic with same name


            for i in range(number_of_same_pic):
            #while count < 65:
                x1 = csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[i,2]
                y1 = csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[i,3]
                x2 = csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[i,4]
                y2 = csv_file[csv_file.iloc[:,0] == full_pic_name].iloc[i,5]

                # print box with coordinates above
                img = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            
            img = cv.copyMakeBorder(img, 40, 40, 40, 40, cv.BORDER_REPLICATE)

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

            #kps = np.array([key for key in keypoints])
            #print(kps)

            for z in range(len(keypoints)):
                if count < 64:
                    new_x1 = int(round(keypoints[z].pt[0])-40)
                    new_y1 = int(round(keypoints[z].pt[1])-40)

                    new_x2 = int(round(keypoints[z].pt[0])+40)
                    new_y2 = int(round(keypoints[z].pt[1])+40)
                    print(keypoints[z].pt[0], keypoints[z].pt[1])
                    cropped_im = img[new_y1:new_y2, new_x1:new_x2]

                    print(count)
                    plt.imshow(cropped_im)
                    plt.show()
                    #return image
                    count += 1
