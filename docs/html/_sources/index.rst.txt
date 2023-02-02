Project 1: Schrodinger - The Crater Detection Algorithm
=====================================================

Synopsis:
---------

Impact craters are the most ubiquitous surface feature on rocky planetary bodies. 
Crater number density can be used to estimate the age of the surface: 
the more denselycratered the terrain, the older the surface. 

When independent absolute ages for a surface are available for calibration of crater counts, 
as is the case for some lava flows and regions of the Moon, crater density can be used to estimate an absolute age of the surface.

Crater detection and counting has traditionally been done by laborious manual interrogation of images
of a planetary surface taken by orbiting spacecraft (Robbins and Hynek, 2012; Robbins, 2019). 
However,the size frequency distribution of impact craters is a steep negative power-law, 
implying that there are many small craters for each larger one. For example, for each 1-km crater on Mars, 
there are more than a thousand 100-m craters. With the increased fidelity of cameras on orbiting spacecraft, 
the number of craters visible in images of remote surfaces has become so large that manual counting is unfeasible. 
Furthermore, manual counting can be time consuming and subjective (Robbins et al., 2014). 
This motivates the need for automated crater detection and counting algorithms (DeLatte et al., 2019).
Recent work has shown that widely used object detection algorithms from computer vision, 
such as the YOLO (You Only Look Once) object detection algorithm (Redmon et al., 2016; Jocher et al., 2021), 
can be effective for crater detection on Mars (Benedix et al., 2020; Lagain et al., 2021) and the Moon (Fairweather et al., 2022).

This tool provides the yolov5 models used for detecting a wide range of sizes of crater both on Mars and on Moon dataset.

Model Description
------------------

Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our tool takes input as one or loads of images of the surface of a planet, 
as well as optional inputs of the planet, planet radius, location and physical size of the image. 
Our tool has been tested for Mars and the Moon dataset.

Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our tool output a list of all the bounding boxes for craters detected in each image.

The additional requirements are the following:

Generate a visualisation of the original image with annotated bounding boxes.

When a real-world image size and location is provided for the image, 
our tool can also provide physical locations and size for each crater.

When ground truth labels are provided, our tool should also determine the number of 
True Positive, False Positive and False Negative detections and return these values for each image.

Finally, the tool can also plot a comparison of the ground truth bounding boxes and 
the model detection bounding boxes.

Measure of performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first metric we are using is IOU, which is the intersection over union. IOU is defined as 
the area of union divided by the area of intersection. The aim of this metric is to find the differences 
between ground truth annotations and preds bounding boxes. This metric is widely used in the crater detection 
algorithms. In object detection, the model predicts multiple bounding boxes for each object. What is more, 
based on the confidence scores of each bounding box, it removes unnecessary boxes according to its threshold 
value. In our case, we use the IOU = 0.45 - 0.50 in both training dataset and the validation dataset.

The second metric we are using is mAP. Since we only have one class: craters to focus on, actually it is the
same as the ap, which is the average precision. It is indeedthe weighted mean of Precision scores achieved at 
each PR curve threshold, with the increase in Recall from the previous threshold used as the weight. During training stage, 
both maP50 and maP95 are evaluated.

Visualisation
~~~~~~~~~~~~~~~

Users can visualize the following:

The original input image without labels.

The original input image with bounding boxes for craters detected by the yolov5.

The original input image with bounding boxes for both the results of the yolov5 with 
comparison for the ground truth bounding boxes.

A separate plot of the cumulative crater size-frequency distribution of detected craters.

Performance statistics including the number of True Positive, False Negative and False 
Positive detections.


Additional sections
~~~~~~~~~~~~~~~~~~~


.. Function API
.. ============

.. .. automodule:: locator
..   :members: PostcodeLocator, great_circle_distance, get_sector_code
..   :imported-members: PostcodeLocator, great_circle_distance, get_sector_code

.. .. automodule:: solver
..   :members: Planet
..   :imported-members: Planet

.. .. automodule:: damage
..   :members: damage_zones, impact_risk
..   :imported-members: damage_zones, impact_risk

.. .. automodule:: mapping
..   :members: plot_circle
..   :imported-members: plot_circle

.. .. automodule:: extensions
..   :members: findstrengthradius, plot_against, getfunctionvalue, searchstrength
..   :imported-members: findstrengthradius, plot_against, getfunctionvalue, searchstrength