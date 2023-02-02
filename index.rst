Project 2: MoonShot - The cater detection
=====================================================

Synopsis
---------

Impact craters are the most ubiquitous surface feature on rocky planetary bodies. 
Crater number density can be used to estimate the age of the surface: 
the more densely cratered the terrain, the older the surface. 
When independent absolute ages for a surface are available for calibration of crater counts, 
as is the case for some lava flows and regions of the Moon, 
crater density can be used to estimate an absolute age of the surface.

Crater detection and counting has traditionally been done by laborious manual 
interrogation of images of a planetary surface taken by orbiting spacecraft (Robbins and Hynek, 2012; Robbins, 2019). 
However, the size frequency distribution of impact craters is a steep negative power-law, 
implying that there are many small craters for each larger one. 
For example, for each 1-km crater on Mars, there are more than a thousand 100-m craters. 
With the increased fidelity of cameras on orbiting spacecraft, 
the number of craters visible in images of remote surfaces has become so large that manual counting is unfeasible. Furthermore, 
manual counting can be time consuming and subjective (Robbins et al., 2014). 
This motivates the need for automated crater detection and counting algorithms (DeLatte et al., 2019).


SubTasks
------------------

Crater Detection Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Develop a module for automatically locating craters in images. 

A training dateset for the Moon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Develop a separate CDM for the Moon and process the moon data before
training the model. The images provided are from a global mosaic of
LROC WAC images of the Moon (100 m/px).

The four images provided are for the regions:

A: -180 to -90 longitude, -45 to 0 latitude;
B: -180 to -90 longitude, 0 to 45 latitude;
C: -90 to 0 longitude, -45 to 0 latitude;
D: -90 to 0 longitude, 0 to 45 latitude.

The test images will be taken from somewhere in the region 0 to 180 longitude; -45 to 45 latitude.

A tool for analysis of craters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The final CDMs for the Moon and Mars should be implemented within a single end-to-end data processing tool
that can be used to analyse craters on either planetary surface. To assess the tool, the images of
the Mars and the Moon will be provided and all craters must be detected and be located.

The purpose of this tool is to allow a user to quickly and automatically identify all craters in
the image and from this generate a size-frequency distribution of the craters for the purpose of
dating the planetary surface. 
    

Submodule Descriptions
----------------------

The :mod:`armageddon` package implements two interdependent features:

   1. An airburst solver using a debris cloud model of meteoroid disruption to
      predict burst energy and location from initial trajectory parameters
   2. An airblast damage mapper to identify and plot damage zones, enumerate
      high-risk postcodes in the UK and estimate fiducial risk

The package is divided into four submodules: :mod:`armageddon.solver`,
:mod:`armageddon.damage`, :mod:`armageddon.locator` and
:mod:`armageddon.mapping`.

:mod:`armageddon.solver`
~~~~~~~~~~~~~~~~~~~~~~~~

Implements the :class:`armageddon.solver.Planet` class for instantiating
planetary parameters with methods for predicting airburst or cratering events
and determining their explosion energies from initial trajectories using a
fourth-order Runge-Kutta ODE solver.

cf. :func:`armageddon.solver.Planet.solve_atmospheric_entry`,
:func:`armageddon.solver.Planet.calculate_energy`,
:func:`armageddon.solver.Planet.analyse_outcome`

:mod:`armageddon.damage`
~~~~~~~~~~~~~~~~~~~~~~~~

Calculates the explosion centre and damage zone radii in geocentric coordinates
given the meteoroid entry parameters. Performs a simple but actionable risk
analysis for emergency and rescue organisations using perturbations in entry
parameters averaged over census population data.

cf. :func:`armageddon.damage.damage_zones`,
:func:`armageddon.damage.impact_risk`

:mod:`armageddon.locator`
~~~~~~~~~~~~~~~~~~~~~~~~~

Extracts census population data by postcode sectors and units in the UK, and
returns population in concentric circles of damage level radii around the
explosion centre. Implements the :class:`armageddon.locator.PostcodeLocator`
class which successively filters for postcodes using bounding box and great
circle distance functions.

cf. :func:`armageddon.locator.great_circle_distance`,
:func:`armageddon.locator.PostcodeLocator.get_postcodes_by_radius`,
:func:`armageddon.locator.PostcodeLocator.get_population_of_postcode`

:mod:`armageddon.mapping`
~~~~~~~~~~~~~~~~~~~~~~~~~

Includes a utility function to generate an interactive map with damage zones
classified according to overpressure levels.

cf. :func:`armageddon.mapping.plot_circle`

Function API
============

.. automodule:: armageddon

.. automodule:: armageddon.solver

.. autoclass:: armageddon.solver.Planet
  :members: solve_atmospheric_entry, calculate_energy, analyse_outcome

.. automodule:: armageddon.damage
   :members: damage_zones, impact_risk

.. automodule:: armageddon.locator
   :members: great_circle_distance

.. autoclass:: armageddon.locator.PostcodeLocator
   :members: get_postcodes_by_radius, get_population_of_postcode

.. automodule:: armageddon.mapping
   :members: plot_circle
