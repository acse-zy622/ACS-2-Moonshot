# Moonshot
The aim of this project is to develop a software tool for
automatically detecting impact craters in images of planetary surfaces
and deriving from this a crater-size frequency distribution that can
be used for dating.

## Table of Contents

- [Introduction](#introduction)
- [Install](#install)
- [Usage](#usage)
- [Tool instruction](#tool_instruction)
- [License](#license)

## Introduction

Impact craters are the most ubiquitous surface feature on rocky
planetary bodies. Crater number density can be used to estimate the
age of the surface: the more densely cratered the terrain, the older
the surface. When independent absolute ages for a surface are
available for calibration of crater counts, as is the case for some
lava flows and regions of the Moon, crater density can be used to
estimate an absolute age of the surface.

Crater detection and counting has traditionally been done by laborious
manual interrogation of images of a planetary surface taken by
orbiting spacecraft
([Robbins and Hynek, 2012](https://doi.org/10.1029/2011JE003966);
[Robbins, 2019](https://doi.org/10.1029/2018JE00559)). However,
the size frequency distribution of impact craters is a steep negative
power-law, implying that there are many small craters for each larger
one. For example, for each 1-km crater on Mars, there are more than
a thousand 100-m craters. With the increased fidelity
of cameras on orbiting spacecraft, the number of craters visible in
images of remote surfaces has become so large that manual counting is
unfeasible. Furthermore, manual counting can be time consuming and
subjective
([Robbins et al., 2014](https://doi.org/10.1016/j.icarus.2014.02.022)).
This motivates the need for automated crater detection and counting algorithms ([DeLatte et al., 
2019](https://doi.org/10.1016/j.asr.2019.07.017)).

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

### Locating the craters
Using one or more images of a planet's surface as well as data such as the radius of the planet as input, 
the image of the impact crater on the planet's surface and the physical location 
of the impact crater (latitude, longitude, and size of the crater at the center) can be located and exported.

### Visualization
For visualization, the following functions can be implemented:
1. The original input image without annotations
2. The original input image with bounding boxes for craters detected by the CDM
3. The original input image with bounding boxes for both the results of the CDM and for the ground truth bounding boxes, if available
4. A separate plot of the cumulative crater size-frequency distribution of detected craters,  if information to calculate crater size is provided
If ground truth data is available, performance statistics including the number of True Positive,  False Negative and False Positive detections.

## Tool instruction

### Mars model
yolov5

### Moon model
yolov5

### Visulisation

## License
