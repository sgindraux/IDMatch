# -*- coding: utf-8 -*-
###
# IDMatch is run with this python file.
# This python script reads the parameter file filled by the user (params.py) and calls functions from functions.py
###

## Import functions and parameter files
from idmatch import functions
from idmatch import params
import shutil

## Create temporary file:
result_path, result_inputdata, result_matching, result_postfiltering, result_plots = functions.create_tempfile(params.result_folderpath)

## Check user inputs from parameter file
parameters_table = functions.check_userinputs(params.mode, params.Image1, params.Image2, params.DSM1, params.DSM2,
                                              params.img_filter_list, params.dsm_filter_list, params.hil_filter_list,
                                              params.pixel_dev, params.step_points, params.nbr_windows,
                                              params.method_list, params.times_step, params.min_window_members,
                                              params.validation_pts_path, params.result_folderpath,
                                              params.gl_extent_path)

## START MAIN WORKFLOW
print("Start of IDMatch in mode = ", params.mode)

if params.mode == 1:
    print("Image matching, using ", params.Image1, "and ", params.Image2, " as image pair")

    # 1. Import images
    mask_input, input_information = functions.import_img(params.Image1, params.Image2, result_inputdata, result_plots, params.gl_extent_path, params.validation_pts_path)

    # 2a. Pre-filter images
    if params.img_filter_list != []: functions.prefilter_img(params.img_filter_list, result_inputdata, mask_input)

if params.mode == 2:
    print("DSM matching, using ", params.DSM1, "and ", params.DSM2, " as DSM pair")

    # 1. Import DSMs
    mask_input, input_information = functions.import_dsm(params.DSM1, params.DSM2, result_inputdata, result_plots, params.gl_extent_path, params.validation_pts_path)

    # 2a. Pre-filter DSMs
    if params.dsm_filter_list != []: functions.prefilter_dsm(params.dsm_filter_list, result_inputdata)

    # 2b. Pre-filter Hillshades
    if params.hil_filter_list != []: functions.prefilter_hil(params.hil_filter_list, result_inputdata, mask_input)

if params.mode == 3:
    print("Image and DSM matching, using ", params.Image1, " and ", params.Image2, " as image pair, as well as ", params.DSM1, " and ", params.DSM2, " as DSM pair")

    # 1. Import images and DSMs
    mask_input, input_information = functions.import_datasets(params.Image1, params.Image2, params.DSM1, params.DSM2, result_inputdata, result_plots, params.gl_extent_path, params.validation_pts_path)

    # 2a. Pre-filter images
    if params.img_filter_list != []: functions.prefilter_img(params.img_filter_list, result_inputdata, mask_input)

    # 2b. Pre-filter DSMs
    if params.dsm_filter_list != []: functions.prefilter_dsm(result_inputdata, params.dsm_filter_list)

    # 2c. Pre-filter Hillshades
    if params.hil_filter_list != []: functions.prefilter_hil(result_inputdata, params.hil_filter_list, mask_input)

# 3. Match images/DSMs/Hillshades
tot_matching_pts = functions.matching(result_inputdata, result_matching, params.method_list, mask_input, params.step_points, params.pixel_dev, params.nbr_windows, input_information)

# 4. Post-filter
table_mean_results, table_postfiltering, table_validation1, table_values_combi = functions.postfilter(params.validation_pts_path, result_path, result_matching, result_postfiltering, params.step_points, params.min_window_members, params.times_step, input_information, tot_matching_pts)

# 5. Plot results
functions.display_results(result_postfiltering, result_plots, table_mean_results, input_information, table_postfiltering, params.mode, table_validation1, params.validation_pts_path)


## Terminating:
# Remove all files from temporary folder
shutil.rmtree(result_matching)  # Remove temporary matching folder

print("IDMatch terminated: The results can be found in ", result_path)
