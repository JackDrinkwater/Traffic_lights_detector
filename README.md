# Traffic lights detector

# 0. Annotation 

This repository is declared to realizing the method of traffic lights detecting which is performed in this [article]. Described method based on only vision system that is often attampted to be implemented in autonomous driving systems and advanced driver-assistance systems. Such systems use only input images from a camera and no other sensors like radar or lidar. One of important tasks for sush system is locacalization of traffic lights and then understanding them for making a choice by,for example, autonomous vehicle. 

The repository implements the approach, proposed by authors of the article, of a traffic light recognition system "where adaptive thresholding and deep learning are used for region proposal and traffic light localization, respectively". The LISA open-source dataset is used along with custom augmentation methods in order to increase the number of available data samples. Authors declare that "the classification part of the algorithm gives a total of 89.60% true detection rate, while the regression part of the model produced a correct location of the traffic light in 92.67% of cases".

# 1. Data

Data can be loaded [here](https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset/download). The database is collected in San Diego, California, USA. The database provides four day-time and two night-time sequences primarily used for testing, providing 23 minutes and 25 seconds of driving in Pacific Beach and La Jolla, San Diego. Capture images with a resolution of 1280 x 960. The training clips consists of 13 daytime clips and 5 nighttime clips.

<p align="center">
Examples of a day and a night frame:
</p>

<img src="https://github.com/JackDrinkwater/Traffic_lights_detector/blob/main/pics/dayClip5--00005.jpg" width="450" height="300"> <img src="https://github.com/JackDrinkwater/Traffic_lights_detector/blob/main/pics/nightClip5--00005.jpg" width="450" height="300">

The annotations are stored as 1 annotation per line with the addition of information such as class tag and file path to individual image files. With this structure the annotations are stored in a csv file.

# 2. Use

### Requirements:
    - Tensorflow 2.x
    - OpenCV     4.x
    - Numpy      1.16.x
    - Pandas     
    - Scikit-learn
 
### Starting with:

```
git clone https://github.com/JackDrinkwater/Traffic_lights_detector
```
The next is loading the data and putting it in folder "data".

### Training:

For training the model in terminal go to "src/" folder and then:
```
python train.py
```

### Example:

For testing the model on one image go to "src/" folder and then:
```
python example_test.py
```
