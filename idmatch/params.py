### params.py ###
# This file contains all parameters that the user of IDmatch needs to define. The parameter file will feed the core.py (the main code) together with the functions.py (the functions used in the main code).
# Only this file needs to be adapted by the user! The rest (core.py and functions.py) stay untouched!

## There are three mode in the IDMatch software. Choose one mode (e.g. mode = 2)
# Mode 1: Two ortho-images are matched
# Mode 2: Two DSMs are matched
# Mode 3: Two ortho-images and DSMs are matched (separately) and the results are merged together

mode = 1
# --> Once your mode chosen, please fill all parameters related to your mode only (leave the others as they are) and fill the common parameters !!!

#------------ MODE 1 ------------------
#--------------------------------------

###--- 1. Insert image file paths
Image1 = "C:\\Users\\Gindraux\\Documents\\github\\IDMatch\\tests\\data_image\\Gries\\20160815_Generic_Mosaic_cc_05m.tif" # Path to the first image (for the moment, must be .tif).
Image2 = "C:\\Users\\Gindraux\\Documents\\github\\IDMatch\\tests\\data_image\\Gries\\20160913_Generic_Mosaic_cc_05m.tif" # Path to the second image (for the moment, must be .tif)

###--- 2a. The images can be pre-filtered (i.e. go through some noise filter or image enhancement). The user can define which ones to apply.
# If you do not want your DSMs to be pre-filtered, leave the list empty (img_filter_list = []). All possible combinations: img_filter_list = ['F1', 'F2', 'F3', 'F1F2', 'F1F3']
# F1: Median filter
# F2: Local histogram equalization
# F3: Contrast Limited Adaptive Histogram Equalization (CLAHE).
img_filter_list = []


#------------ MODE 2 ------------------
#--------------------------------------

###--- 1. Insert DSM file paths
DSM1 = "C:\\Users\\Gindraux\\Documents\\github\\IDMatch\\tests\\data_dsm\\Gries\\20160815_DSM_05m_scaled.tif"  # Path to the first digital surface model (DSM) (for the moment, must be .tif)
DSM2 = "C:\\Users\\Gindraux\\Documents\\github\\IDMatch\\tests\\data_dsm\\Gries\\20160913_DSM_05m_scaled.tif"  # Path to the second digital surface model (DSM) (for the moment, must be .tif)

###--- 2. The DSMs will be used in two different ways. 1) as height model 2) as hillshade image. The filters for both type of data are therefore different and need to be separatedly specified

##-- 2a. f you do not want your DSMs to be pre-filtered, leave the list empty (dsm_filter_list = []). All combinations are dsm_filter_list = ['F1', 'F4', 'F1F4']
# F1: Median filtering
# F4: Bilateral filtering (edge-preserving)
dsm_filter_list = ['F1', 'F4', 'F1F4']

##-- 2b. If you do not want your Hillshades to be pre-filtered, leave the list empty (hil_filter_list = []). All combinations are hil_filter_list = ['F1', 'F5', 'F1F5']
# F1: Median filter
# F5: Feature Canny (Edge detection)
hil_filter_list = ['F1', 'F5', 'F1F5']


#--------- Common parameters ----------
#--------------------------------------

###--- 3. Chose the matching methods you want to apply on your dataset
# M1: Normalized Cross Correlation (NCC)
# M2: Orientation Correlation (OC)
# M3: Feature matching with SURF and FLANN (M3 is not working for the DSMs. If mode=2 is chosen, only the hillshades will be used as input to this method)
method_list = ['M1']  # it is possible to choose several for one IDMatch run method_list = ['M1', 'M2', 'M3']

pixel_dev = 6  # 6,8, 15,  12 # integer. estimated deviation or velocity (in pixel) between the input data pair. It will be used to define the window sizes in the matching function (matching)
step_points = 50  # integer. in pixels
nbr_windows = 1  # integer. number of different window sizes that will be used for the matching iterations.

###--- 4. Post-filtering
times_step = 8  # must be odd and not zero (wind_size = step_points * times_stepgrid)
min_window_members = 15  # integer.

###--- 5. Others
##-- 5a. Do you have validation points? If yes, insert path. If not, leave empty (validation_pts_path = "")
# The table need to have one line headers, with four columns x1, y1, x2, y2 (coordinates x and y at the first and second epoch), tab delimited.
validation_pts_path = "C:\\Users\\Gindraux\\Documents\\github\\IDMatch\\tests\\validation_pts\\stakes_gries.txt"

##-- 5b. Do you have a glacier extent? If yes, insert path. If not, leave empty (gl_extent_path = "")
gl_extent_path = "C:\\Users\\Gindraux\\Documents\\github\\IDMatch\\tests\\glacier_extent\\extent_gries.txt"

##-- 5c. Set path where you want to save the software's outputs (results)
result_folderpath = "X:\\sgindraux\\Results_IDMatch\\AGM2_plots_extent"  # result_folderpath = ".\\tests"


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
