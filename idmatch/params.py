### params.py ###
# This file contains all parameters that the user of IDmatch needs to define. The parameter file will feed the core.py (the main code) together with the functions.py (the functions used in the main code).
# Only this file needs to be adapted by the user! The rest (core.py and functions.py) stay untouched.

## There are three mode in the IDMatch software. Choose one mode (e.g. mode = 2)
# Mode 1: Two ortho-images are matched
# Mode 2: Two DSMs are matched
# Mode 3: Two ortho-images and DSMs are matched (separately) and the results are merged together

mode = 1
# --> Once your mode chosen, please fill all parameters related to your mode only (leave the others as they are) and fill the common parameters !!!

#*** Run tests ***
# To run the test case of Gries glacier:
# - change the paths below based on the location of the IDMatch folder.
# - de-zip the DSMs files
# ***

#------------ MODE 1 ------------------
#--------------------------------------

###--- 1. Insert image file paths
Image1 = "C:\\Users\\Gindraux\\Documents\\IDMatch\\tests\\20160815_Gries_img_1m.tif" # Path to the first image (must be .tif)
Image2 = "C:\\Users\\Gindraux\\Documents\\IDMatch\\tests\\20160913_Gries_img_1m.tif" # Path to the second image (must be .tif)

###--- 2a. The images can be pre-filtered (i.e. go through some noise filter or image enhancement). The user can define which ones to apply.
# If you do not want your DSMs to be pre-filtered, leave the list empty (img_filter_list = []). The default is all possible combinations: img_filter_list = ['F1', 'F2', 'F3', 'F1F2', 'F1F3']
# F1: Median filter
# F2: Local histogram equalization
# F3: Contrast Limited Adaptive Histogram Equalization (CLAHE).
img_filter_list = ['F1', 'F2', 'F3', 'F1F2', 'F1F3']


#------------ MODE 2 ------------------
#--------------------------------------

###--- 1. Insert DSM file paths
DSM1 = "C:\\Users\\Gindraux\\Documents\\IDMatch\\tests\\20160815_Gries_dsm_1m.tif"  # Path to the first digital surface model (DSM) (must be .tif)
DSM2 = "C:\\Users\\Gindraux\\Documents\\IDMatch\\tests\\20160913_Gries_dsm_1m.tif"  # Path to the second digital surface model (DSM) (must be .tif)

###--- 2. The DSMs will be used in two different ways. 1) as height model 2) as hillshade image. The filters for both type of data are therefore different and need to be separatedly specified

##-- 2a. f you do not want your DSMs to be pre-filtered, leave the list empty (dsm_filter_list = []). The default is all combinations: dsm_filter_list = ['F1', 'F4', 'F1F4']
# F1: Median filtering
# F4: Bilateral filtering (edge-preserving)
dsm_filter_list = ['F1', 'F4', 'F1F4']

##-- 2b. If you do not want your Hillshades to be pre-filtered, leave the list empty (hil_filter_list = []). The default is all combinations: hil_filter_list = ['F1', 'F5', 'F1F5']
# F1: Median filter
# F5: Feature Canny (Edge detection)
hil_filter_list = ['F1', 'F5', 'F1F5']


#--------- Common parameters ----------
#--------------------------------------

###--- 3. Chose the matching methods you want to apply on your dataset
# M1: Normalized Cross Correlation (NCC)
# M2: Orientation Correlation (OC)
# M3: Feature matching with SURF and Brute Force Matcher
method_list = ['M1', 'M2', 'M3']  # it is possible to run IDMatch with individual methods. The default is: method_list = ['M1', 'M2', 'M3']

pixel_dev = 6  # integer, estimated deviation or velocity (in pixel) between the input data pair. It will be used to define the window sizes in the matching function (matching)
step_points = 25  # integer, in pixels
nbr_windows = 5  # integer, number of different window sizes that will be used for the matching iterations.

###--- 4. Post-filtering
times_step = 8  # must be odd and not zero (wind_size = step_points * times_stepgrid)
min_window_members = 15  # integer.

###--- 5. Others
##-- 5a. Do you have validation points of displacement? If yes, insert path to your .txt file. If not, leave empty (validation_pts_path = "")
validation_pts_path = "C:\\Users\\Gindraux\\Documents\\IDMatch\\tests\\stakes_gries.txt"  # The table need to have one line headers, with four columns x1, y1, x2, y2 (coordinates x and y at the first and second epoch), tab delimited.


##-- 5b. Do you want to run IDMatch on a specific area of the image/dsm? If yes, insert path to your .txt file. If not, leave empty (gl_extent_path = "")
gl_extent_path = "C:\\Users\\Gindraux\\Documents\\IDMatch\\tests\\extent_gries.txt"  # The table need to have one line headers, with 2 columns x, y (coordinates of points delineating the object extent / area where the processing takes place), tab delimited.


##-- 5c. Set path where you want to save the software's outputs (results)
result_folderpath = "C:\\Users\\Gindraux\\Documents\\IDMatch\\tests\\results_idmatch"


#------------ Advanced parameters ------------------

# Several additional parameters can be changed in the functions.py file. They are listed below and can be found in the function.py file by searching for " !!!advanced_param: "

###--- In Matching (def matching)
# wsize_min, wsize_max
# win_subpix

###--- In post-filtering (def postfilter)
# std_nbr_magn
# std_nbr_dir
# angle_max
# percent_neighbours
# snr_threshold
