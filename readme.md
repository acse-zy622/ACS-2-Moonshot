# Moonshot
The purpose of this project is to build a software tool capable of 
automatically detecting impact craters in planetary surface images 
and producing a crater-size frequency distribution for dating.

## Table of Contents

- [Introduction](#introduction)
- [Install](#install)
- [Usage](#usage)
- [User instructions](#user_instructions)
- [License](#license)

## Introduction

Impact craters are the most common surface feature found on rocky planetary bodies. 
The density of craters can be utilized to determine the age of the surface, 
with a denser terrain indicating an older surface. When there is access to 
independent absolute ages for calibration, crater density can be used to estimate the exact age of the surface.

Recent work has shown that widely used object detection algorithms
from computer vision, such as the YOLO (You Only Look Once) object
detection algorithm ([Redmon et al., 2016](https://doi.org/10.1109/CVPR.2016.91);
[Jocher et al., 2021](https://doi.org/10.5281/zenodo.4418161)), can be effective for crater detection on Mars
([Benedix et al., 2020](https://doi.org/10.1029/2019EA001005); [Lagain et al., 2021](https://doi.org/10.1029/2020EA001598)) and the Moon ([Fairweather
et al., 2022](https://doi.org/10.1029/2021EA002177)).

## Install

package needed
```sh
$ pip install ...
```

## Usage

The aim of this tool is to enable users to efficiently and automatically 
locate all craters in an image, and from this, create a frequency distribution of 
crater sizes for dating the planetary surface. The tool must therefore possess the 
capability to calculate the actual size and location of craters in real-world units 
if the image's location, size, and resolution are supplied.


### Locating the craters
Using one or more images of a planet's surface as well as data such as the radius of the planet as input, 
the image of the impact crater on the planet's surface and the physical location 
of the impact crater (latitude, longitude, and size of the crater at the center) can be located and exported.

### Visualization
For visualization, the following functions are implemented:
1. The initial image without any labels or annotations
2. The initia input image with bounding boxes for craters detected by the CDM
3. The original input image with bounding boxes for both the results of the CDM 
and for the ground truth bounding boxes, if available
4. A standalone graph of the cumulative frequency distribution of the detected 
craters' sizes, if the information required for size calculation is provided.
5. If ground truth data is present, performance metrics including the count of 
True Positive, False Negative, and False Positive detections.

## User instructions
### Overview
To use this tool, first input the ```test_path``` and ```output_path```. Than choose the ```Planet``` : Mars or Moon. Then we can visualize the analysis part:
1. Show the original images. In the ```original image``` option, splitted images (416 pixels) of different regions of the planet are displayed.
2. Show craters detected with Crater Detection Model (CDM). Click the ```CDM Detection``` button, you can display the CDM results on the original images, with the red box circled the monitored craters area.
3. Compare the results. To verify the accuracy of the CDM model, click the ```Results Comparison``` button and locate the position of the known craters with a yellow box on the image that has shown the monitoring results.
4. Show the size-frequency distribution. Click ```S-F Plot``` to show a graph of the cumulative frequency distribution of crater sizes for the detected craters.
5. Performance statistics. Click ```Statistics Analysis``` to show the performance statistics data including the number of True Positive, False Negative and False Positive detections.

### Mars model
a module for automatically locating craters in Mars surface images

### Moon model
a module for automatically locating craters in moon surface images


## License
