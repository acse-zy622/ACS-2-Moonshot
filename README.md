# Moonshot: Automatic Impact Crater Detection on the Moon

<a href="url"><img src="https://drive.google.com/uc?export=view&id=1dJjw6g_S8s5hMsiZ67Sp9f50NrgZvoTm" align="left" height="300" width="300" ></a>

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

## Introduction

This crater tool provides a method to analyse craters on both Mars and the Moon. 

The tool allows the user to quickly and automatically identify all craters in the image and generate a size-frequency distribution of the craters for the purpose of dating the planetary surface.

### Overview

- What model is used for mars?
- What model is used for moon?

### Major Features 

* For each input image, generate a csv file of crater locations.
* For each input image, generate an image with bounding boxes of the craters detected by the model and ground truth bounding boxes, if available. 
* If the user provides information to calculate crater size, generate a plot of the cumulative crater-size frequency distribution of detected craters. 
* If ground truth is available, generate performance statistics including number of True Positive, False Negative and False Positive detections. 

## Installation Guide 

### Prerequisite 

Package manager? Conda?

### Installation and configuration

1. Clone the repository:

`git clone https://github.com/ese-msc-2022/acds-moonshot-schrodinger.git`

2. Go to the git repository on your local computer:

`cd acds-moonshot-schrodinger`

3. Environment activation etc....

## User Instructions 

- Guide on how to use the user interface. 

## License

This project is released under the MIT license. 

## Testing 

The tool includes several tests which can be used to check its operation on your system. 

With pytest installed these can be run with:

`add how to run tests`