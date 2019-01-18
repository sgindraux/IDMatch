.. _label-docs:

Documentation
*************

In summary:

The software has three main files, called:

 * core.py
 * functions.py
 * params.py.

The core.py is the main file, which includes the information of the params.py and functions.py. To run the software, the user need to run the core.py file
The functions.py is a library of functions called from the core.py. The description of the different functions are in :ref:`label-functions`.
The params.py is the file where the paths to the input data as well as other variables are defined. Details about the parameter file can be found in the :ref:`label-params`.That's the only file that the user needs to change.

The input data are an image or a DSM pair. In option, the user can also input validation points.
The output data are:

* XXXX.py: A file with the name of the best combination is returned. Its contenten is: x1 y1 dx2 dy2 peakCorr meanAbsCorr SNratio magn angle val_magn val_angle pts_filtered

* table_results_postfilt.txt: A table with the number of discared points (in percent) from the different post-filters.

* IDMatch_parameters.txt: A table that copies the parameters set by the user in the params.py file.

* velocity_report.pdf: A report displaying the velocity field from the best combination found (best match) and summarizes the results obtained from all iterations.


.. _label-params:

Input parameters
================

In this section, the parameters contained in the params.py file are explained.

Parameters in mode 1 (image matching):

Image1 = "path to your first image", str
Image2 = "path to your second image", str

If you want to pre-filter your input images, you can choose single or combinations of filters (i.e. not all combinations are possible because they don't make sense).
The implemented filters are:
* F1: Median filter
* F2: Local histogram equalization
* F3: Contrast Limited Adaptive Histogram Equalization (CLAHE).

img_filter_list = ['F1', 'F2', 'F3', 'F1F2', 'F1F3'], list of string. This is the list of filters with all implemented possibilities that will be applied to the input images. This is the default list.
If you don't want to pre-filter your images, leave the list empty (img_filter_list = []).


Parameters in mode 2 (DSM matching):

DSM1 = "path to your first DSM", str
DSM2 = "path to your second DSM", str

If you want to pre-filter your input DSMs, you can choose single filters.
The implemented filters are:
* F1: Median filtering
* F2: Bilateral filtering (edge-preserving)

dsm_filter_list = ['F1', 'F2'], list of string. This is the list of filters with all implemented possibilities that will be applied to the input images. This is the default list.
If you don't want to pre-filter your DSMs, leave the list empty (dsm_filter_list = []).

Your DSMs will also be transformed in hillshade images within the software. Similarly, you can pre-filter them.
* F1: Median filter
* F2: Feature Canny (Edge detection)

hil_filter_list = ['F1', 'F2', 'F1F2'], list of string. This is the list of filters with all implemented possibilities that will be applied to the input images. This is the default list.
If you don't want to pre-filter your hillshades, leave the list empty (hil_filter_list = []).



Parameters included in both modes (common parameters):

pixel_dev = 5  # Estimated deviation or velocity (in pixel) between the input data pair. It should be an integer value. It will be used to define the window sizes in the matching function (matching_img).
step_points = 50  # The matching functions will define a template and a search window at defined grid points. step_points indicates the spacing (in pixels) that occur between these points. The denser the points, the longer the computation time. It should be an integer value.
nbr_windows = 5  # This parameter defines the number of different window sizes that will be used for the matching. The larger the number of tested window sizes, the longer the computation time. We recommend to have 3 as minimum, in order to observe the differences in the results. 5 would be the default value and 10 is the ideasl case. It should be an integer value.


If the user has validation points, one can insert the path to the table. If not, leave the list empty (validation_pts_path = "")
validation_pts_path = "path to your validation points table", str

The table should be built as follow:
One line of headers
Four columns, indicating the x and -y coordinates (i.e. easting, northing) of the validation points at time 1 (x1, y1) and the same for time 2 (x2, y2).
The table must be Tab delimited.


Example:
x1	y1	x2	y2
2670526.4159	1144842.9077	2670526.323	1144842.996
2670198.5376	1144897.7678	2670198.583	1144897.847
2670418.5610	1144737.0538	2670418.576	1144737.077


result_folderpath = "path to your desired result folder", str




.. _label-functions:

Functions in IDMatch
====================

There are several functions used in IDMatch and they were built in a way that each step of the workflow (see details in :ref:`label-core`) represents one function.
For instance, one function is about image filtering, another one for image matching and another one for displaying the results.
Only in one case (the matching function) needs to call another one, mainly for the purpose of readability.

In the following, the functions a listed together with their purpose, inputs and outputs.

.. automodule:: test_docstring
    :members:


.. _label-core:

The core file
=============

The core file is the main file collecting the information from the parameters and the functions.

For mode 1, the script calls the functions as follow:

1. Import images: functions.import_img()
2. Pre-filter images: functions.prefilter_img()
3. Match images: functions.matching_img()
4. Post-filter: functions.postfilter_img()
5. Find best option: functions.best_results()
6. Plot results: functions.display_results()

For mode 2, the script calls the functions as follow:

1. Import DSMs: functions.import_dsm()
2. Pre-filter DSMs: functions.prefilter_dsm()
3. Match DSMs: functions.matching_img()
4. Post-filter: functions.postfilter_img()
5. Find best option: functions.best_results()
6. Plot results: functions.display_results()