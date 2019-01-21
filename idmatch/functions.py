## Import libraries
import matplotlib
matplotlib.use('Agg')
from osgeo import gdal, gdalconst
import os
import warnings
import numpy as np
import skimage
from skimage import data, exposure, io, filters
from skimage.color import rgb2gray, rgba2rgb
import cv2
import datetime
from scipy.signal import fftconvolve
from scipy.interpolate import griddata
import shutil
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
import scipy
from PIL import Image, ImageDraw
from scipy import ndimage
from pylab import *


def check_userinputs(mode, image1, image2, dsm1, dsm2, image_filter_list, digitsm_filter_list, hillshade_filter_list, pix_dev, step_grid, nbr_win, method_list, times_stepgrid, min_win_members, val_pts_option, result_path, gl_extent):
    """
    This function checks all input parameters from the params.py file before going through the other functions.
    This way, the workflow does not break later.

    Parameters
    ----------
    all parameters defined in params.py

    Returns
    -------

    table_parameters: .txt file
        File with a copy of the parameters used for the software's run

    """

    ####-------------------------------------------------- MODE 1 --------------------------------------------------------------
    if mode == 1 or mode == 3:

        # 1. Check if image path filled
        if not isinstance(image1, str) or result_path == "": raise Exception('The path to the image1 is either not a string or is empty')
        if not isinstance(image1, str) or result_path == "": raise Exception('The path to the image2 is either not a string or is empty')

        # 2. Check if the filter or filter combination exist
        if image_filter_list != []:
            for filters in image_filter_list:
                if 'F1' or 'F2' or 'F3' or 'F1F2' or 'F1F3' or 'F2F3' in filters:
                    pass
                else: raise Exception('The image filter(s) you selected is either not existing, or has the wrong combination. Please look at the documentation')


    ####-------------------------------------------------- MODE 2 --------------------------------------------------------------
    if mode == 2 or mode == 3:

        # 1. Check if dsm path filled
        if not isinstance(dsm1, str) or dsm1 == "": raise Exception('The path to the dsm1 is either not a string or is empty')
        if not isinstance(dsm2, str) or dsm2 == "": raise Exception('The path to the dsm2 is either not a string or is empty')

        # 2. Check if the filter or filter combination exist
        if digitsm_filter_list != []:
            for filters in digitsm_filter_list:
                if 'F1' in filters or 'F4' in filters or 'F1F4' in filters:
                    pass
                if 'F2' in filters or 'F3' in filters or 'F5' in filters:
                    raise Exception('The dsm filter(s) you selected is either not existing, or has the wrong combination. Please look at the documentation')

        if hillshade_filter_list != []:
            for filters in hillshade_filter_list:
                if 'F1' in filters or 'F5' in filters or 'F1F5' in filters:
                    pass
                if 'F2' in filters or 'F3' in filters or 'F4' in filters:
                    raise Exception('The hillshade filter(s) you selected is either not existing, or has the wrong combination. Please look at the documentation')

    ####-------------------------------------------------- Common variables --------------------------------------------------------------

    # 3. Check the method list:
    if method_list == [] or method_list == ['']: raise Exception('You need to select at least one matching method')
    for methods in method_list:
        if 'M1' in methods or 'M2' in methods or 'M3' in methods:
            pass
        if 'M4' in methods or '':
            raise Exception('The matching method you selected is not existing. Please look at the documentation')

    # 3.a and 3b. Check the matching method parameters:
    if isinstance(pix_dev, int) == False: raise Exception('The pixel_dev is not an integer number')
    if isinstance(step_grid, int) == False: raise Exception('The step_points is not an integer number')
    if isinstance(nbr_win, int) == False: raise Exception('The nbr_windows is not an integer number')
    if isinstance(times_stepgrid, int) == False: raise Exception('The value for times_step is not an integer number')
    if times_stepgrid == 0 or times_stepgrid % 2 != 0: raise Exception('The value for times_step cannot be zero or even. Possible values are: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20')
    if isinstance(min_win_members, int) == False: raise Exception('The min_win_members is not an integer number')

    # 4. Check validation points path
    if val_pts_option == [] or not isinstance(val_pts_option, str): raise Exception('The path to the validation points is either not a string or is empty')

    # 5. Check glacier extent path
    if gl_extent == [] or not isinstance(gl_extent, str): raise Exception('The path to the glacier extent is either not a string or is empty')

    # 6. Check result path
    if not isinstance(result_path, str) or result_path == "": raise Exception('The path to the result folder is either not a string or is empty')
    # -----------------

    ## Save the parameters in .txt file
    if mode == 1:
        parameters = np.array(['Mode:', 'Image1:', 'Image2:', 'img_filter_list:', 'pixel_dev:', 'step_points:', 'nbr_windows:', 'method_list:', 'times_step:', 'min_window_members:', 'validation_pts_path:', 'result_folderpath:', 'gl_extent_path:'])
        par_content = np.array([mode, image1, image2, ','.join(image_filter_list), pix_dev, step_grid, nbr_win, ','.join(method_list), times_stepgrid, min_win_members, val_pts_option, result_path, gl_extent])
    if mode == 2:
        parameters = np.array(['Mode:', 'DSM1:', 'DSM2:', 'dsm_filter_list:', 'hil_filter_list:', 'pixel_dev:', 'step_points:', 'nbr_windows:', 'method_list:', 'times_step:', 'min_window_members:', 'validation_pts_path:', 'result_folderpath:', 'gl_extent_path:'])
        par_content = np.array([mode, dsm1, dsm2, ','.join(digitsm_filter_list), ','.join(hillshade_filter_list), pix_dev, step_grid, nbr_win, ','.join(method_list), times_stepgrid, min_win_members, val_pts_option, result_path, gl_extent])
    if mode == 3:
        parameters = np.array(['Mode:', 'Image1:', 'Image2:', 'DSM1:', 'DSM2:', 'img_filter_list:', 'dsm_filter_list:', 'hil_filter_list:', 'pixel_dev:', 'step_points:', 'nbr_windows:', 'method_list:', 'times_step:', 'min_window_members:', 'validation_pts_path:', 'result_folderpath:', 'gl_extent_path:'])
        par_content = np.array([mode, image1, image2, dsm1, dsm2, ','.join(image_filter_list), ','.join(digitsm_filter_list), ','.join(hillshade_filter_list), pix_dev, step_grid, nbr_win, ','.join(method_list), times_stepgrid, min_win_members, val_pts_option, result_path, gl_extent])

    table_parameters = result_path + "\\IDMatch_results\\IDMatch_parameters.txt"
    headers = 'Parameter used in IDMatch: User input in the params.py file'
    par_content.astype(str)
    tableparam = np.column_stack([parameters, par_content])
    np.savetxt(table_parameters, tableparam, fmt="%-s", newline='\n', header=headers, delimiter='\t')  # fmt="%-s\t%-s"

    ## For postfiltering, check how many points will be filtered depending on the size of the post-filtering window and the minimum number of members.
    t_stepgrid = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    max_members = [8, 24, 48, 80, 120, 168, 222, 286, 358, 438]

    for i in range(0, len(t_stepgrid)):
        if t_stepgrid[i] == times_stepgrid:
            if min_win_members > max_members[i]:
                raise Exception("It is not possible to have this much members in the post-filtering window. Either increase 'times_step' or decrease 'nbr_members_window'.")
            if (max_members[i]-min_win_members) < 4:  # arbitrary value
                raise Warning("The value for 'min_win_members' is very close to the maximum member than the post-filtering can hold. Not all post-filtering windows will have the maximum members because 1) there are some windows that will overlap with the edge of the matching area and 2) in the case the peak correlation of the matching function in on the window edge, the point is discarded and a nan is returned. We recommend to slightly: increase 'times_step' or decrease 'nbr_members_window'.")

    return table_parameters


def create_tempfile(result_path):
    """
    This function imports the path defined by the user, where the IDMatch results are stored.


    Parameters
    ----------
    folder path : str

    Returns
    -------

    res_path : str
        Path to the result folder '\\IDMatch_results'
    res_input : str
        Path to a folder ('\\input_data') where the input datasets used (original data + filters) are stored.
    res_matching : str
        Path to a folder ('\\matching_tables') where the results from matching are stored. This is a temporary folder and fill be removed at the end of IDMatch workflow.
    res_postfilt : str
        Path to a folder ('\\postfiltering_tables') where the results after the post-filtering procedure are stored.
    res_plots : str
        Path to a folder ('\\plots') where selected plots are saved.

    """

    # Remove previous result folder and files if existing
    if os.path.isdir(result_path + '\\IDMatch_results'):
        shutil.rmtree(result_path + '\\IDMatch_results', ignore_errors=True)
        with warnings.catch_warnings():
            print('There is already a folder called "IDMatch_results" in the result_folderpath. If you re-run this program, the folder will be replaced')

    # Creates the temporary folder and subfolders
    os.makedirs(result_path + '\\IDMatch_results')
    os.makedirs(result_path + '\\IDMatch_results\\input_data')
    os.makedirs(result_path + '\\IDMatch_results\\matching_tables')  # temporary
    os.makedirs(result_path + '\\IDMatch_results\\postfiltering_tables')
    os.makedirs(result_path + '\\IDMatch_results\\plots')

    res_path = result_path + '\\IDMatch_results'
    res_input = result_path + '\\IDMatch_results\\input_data'
    res_matching = result_path + '\\IDMatch_results\\matching_tables'  # temporary
    res_postfilt = result_path + '\\IDMatch_results\\postfiltering_tables'
    res_plots = result_path + '\\IDMatch_results\\plots'

    return res_path, res_input, res_matching, res_postfilt, res_plots


def import_img(image1, image2, res_input, res_plots, gl_extent, val_pts_option):
    """
    This function is called in mode=1 (image mode). It imports the two input images (.tif) as raster. If the images do not have the same cell size, the image with the smaller
    cell size is resampled based on the image with the larger cell size. If the images do not have the same extent, both images are
    saved with their maximum common extent. Both images are saved in greyscale for further processing, as well as in a lower resolution for later plots.

    Parameters
    ----------
    image1 : str
        Path to the first image
    image2 : str
        Path to the second image
    res_input: str
        Path to the input data folder
    res_plots: str
        Path to the plot folder
    gl_extent: str
        Path to the .txt file containing points delineating a processing extent (mask) if available
    val_pts_option: str
        Path to the .txt file containing validation points if available


    Returns
    -------
    mask_data : ndarray
        Two dimensional numpy array with 0 and 1. 1 is foreground of image 1 and image 2 (the intersecting extent between both images) and 0 the background

    image_information : list
        List containing the coordinates of image origin in the x- and y-dimension, the pixel width and height, number of rows and columns of the image as well as the projection. In order: xOrigin, pixelWidth, yOrigin, pixelHeight, rows, cols, proj

    """

    ## Message:
    print("## Importing images")

    ## Check if both image paths were inserted and they have the .tif extention  (# later check for gtif, gtiff, jp2, jpeg2000, or geojp2.  --> im2.lower().endswith(('.png', '.jpg', '.jpeg')))
    if isinstance(image1, str) == False or isinstance(image2, str) == False:
        raise Exception('You need to insert two images')
    if image1.lower().endswith('.tif') == False or image2.lower().endswith('.tif') == False:
        raise Exception('The images file type must be tif!')

    ## Open images in gdal
    img1 = gdal.Open(image1, gdalconst.GA_ReadOnly)
    img2 = gdal.Open(image2, gdalconst.GA_ReadOnly)
    geot_img1 = img1.GetGeoTransform()  # Geotransform: [0] =  top left x, [1] w-e pixel resolution, [3] top left y, [5] n-s pixel resolution (negative value)
    geot_img2 = img2.GetGeoTransform()
    img1_proj = img1.GetProjection()

    ## Find the common area between both images
    ext_img1 = [geot_img1[0], geot_img1[3], geot_img1[0] + (geot_img1[1] * img1.RasterXSize), geot_img1[3] + (geot_img1[5] * img1.RasterYSize)]
    ext_img2 = [geot_img2[0], geot_img2[3], geot_img2[0] + (geot_img2[1] * img2.RasterXSize), geot_img2[3] + (geot_img2[5] * img2.RasterYSize)]
    intersection = [max(ext_img1[0], ext_img2[0]), min(ext_img1[1], ext_img2[1]), min(ext_img1[2], ext_img2[2]), max(ext_img1[3], ext_img2[3])]
    if (intersection[2] < intersection[0]) or (intersection[1] < intersection[3]):raise Exception('There is no intersection between both input images')
    if ((intersection[2] - intersection[0]) / geot_img1[1]) < 100 or ((intersection[1] - intersection[3]) / geot_img1[1]) < 100:
        raise Exception('There is too little intersection (less than 100 x 100 pixels) between both input images')

    # Get larger cell size (if differences)
    cellsize = np.max([geot_img1[1], geot_img2[1]])

    # Save images to the same resolution and extent
    gdalwarp_str_img1 = ''.join(('gdalwarp -overwrite -tr ', str(cellsize), ' ', str(cellsize), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ', str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', image1, ' ', res_input + '\\i1.tif'))  # prepare str for gdalwarp module. Image 1 saved as I1 with intersecting extent (between I1 and I2
    os.system(gdalwarp_str_img1)

    gdalwarp_str_img2 = ''.join(('gdalwarp -overwrite -tr ', str(cellsize), ' ', str(cellsize), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ', str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', image2, ' ', res_input + '\\i2.tif'))  # prepare str for gdalwarp module. Image 2 saved as I2 with cell size of image 1 and intersecting extent
    os.system(gdalwarp_str_img2)

    # Re-import images and store them in greyscale
    new_image1 = skimage.data.load(os.path.abspath(res_input + '\\i1.tif'))
    new_image2 = skimage.data.load(os.path.abspath(res_input + '\\i2.tif'))
    new_img_geotransf = (intersection[0], cellsize, 0.0, intersection[1], 0.0, -cellsize)
    rows, cols = new_image1.shape[:2]  # new_image1 and new_image2 have the same shape

    # Get information of the new images
    image_information = (intersection[0], cellsize, intersection[1], -cellsize, rows, cols, img1_proj) # xOrigin, pixelWidth, yOrigin, pixelHeight, rows, cols, proj

    if len(new_image1.shape) == 2:  # Check whether the uploaded images are already grayscale
        image_gray1 = new_image1
    elif new_image1.shape[2] == 3:  # Check whether the image have 3 channels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # to ignore: UserWarning: Possible precision loss when converting from float64 to uint8
            image_gray1 = skimage.img_as_ubyte(rgb2gray(new_image1))
    elif new_image1.shape[2] == 4:  # Check whether the image have 4 channels
        image_rgb = rgba2rgb(new_image1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # to ignore: UserWarning: Possible precision loss when converting from float64 to uint8
            image_gray1 = skimage.img_as_ubyte(rgb2gray(image_rgb))  # !! There is no direct function rgba2gray (yet)
    else:
        raise Exception('Image 1 is not grayscale, RGB or RGBA')

    if len(new_image2.shape) == 2:
        image_gray2 = new_image2
    elif new_image2.shape[2] == 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image_gray2 = skimage.img_as_ubyte(rgb2gray(new_image2))
    elif new_image2.shape[2] == 4:
        image_rgb = rgba2rgb(new_image2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image_gray2 = skimage.img_as_ubyte(rgb2gray(image_rgb))
    else:
        raise Exception('Image 2 not grayscale, RGB or RGBA')


    ## Create mask in which the pre-filter, matching and postfilter steps (functions) will occur later
    if gl_extent != "":  # if user added a glacier_extent file use this one, otherwise use the common glacier extent calculated above (image_information).
        ext_x, ext_y = np.loadtxt(gl_extent, skiprows=1, unpack=True, delimiter="\t")

        # Transform world coordinates into pixel coordinates (in float)
        ext_px = []; ext_py = []
        for i in range(0, len(ext_x)):
            ext_py.append((intersection[1] - ext_y[i]) / cellsize)
            ext_px.append((ext_x[i] - intersection[0]) / cellsize)

        # Check if extent points are inside the image extent
        count = 1; store = []
        for v_px, v_py in zip(ext_px, ext_py):
            if v_px > cols or v_py > rows:
                store.append(count)
                count = count + 1
            else:
                count = count + 1
        if store != []:
            raise Exception('The point on line number', store, ' of your glacier extent table is outside the image extent')

        # Transform the points in polygon
        polygon = list(zip(ext_px, ext_py))
        imgaage = Image.new('L', (cols, rows), 0)
        ImageDraw.Draw(imgaage).polygon(polygon, outline=1, fill=1)
        mask_data = np.array(imgaage)

    else:
        ## Create intersecting mask (mask = 1, otherwise 0) between both images and apply them
        a = image_gray1 != 255  # takes out white background
        b = image_gray1 != 0  # takes out black background
        a2 = image_gray2 != 255
        b2 = image_gray2 != 0
        mask_data = np.where((a == 1) & (b == 1) & (a2 == 1) & (b2 == 1), 1, 0)

    ## Save the new images (the ones that will be used in further processing) in the temporary file
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(res_input + '\\I1.tif', cols, rows, 1, gdal.GDT_Byte, )
    dataset.SetGeoTransform(new_img_geotransf)
    dataset.SetProjection(img1_proj)
    dataset.GetRasterBand(1).WriteArray(image_gray1)
    dataset.FlushCache()  # Write to disk.

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(res_input + '\\I2.tif', cols, rows, 1, gdal.GDT_Byte, )
    dataset.SetGeoTransform(new_img_geotransf)
    dataset.SetProjection(img1_proj)
    dataset.GetRasterBand(1).WriteArray(image_gray2)
    dataset.FlushCache()

    # Resize the above images that will be used for plotting in a later function
    print("Exporting in lower resolution for plotting")
    resizing = 2  # to 2 m resolution
    gdalwarp_I1 = ''.join(('gdalwarp -overwrite -tr ', str(resizing), ' ', str(resizing), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ',str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', res_input + '\\I1.tif', ' ', res_plots + '\\I1_resize.tif'))
    os.system(gdalwarp_I1)
    gdalwarp_I2 = ''.join(('gdalwarp -overwrite -tr ', str(resizing), ' ', str(resizing), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ', str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', res_input + '\\I2.tif', ' ', res_plots + '\\I2_resize.tif'))
    os.system(gdalwarp_I2)

    # Check whether the validation points (if there are) are inside the image extent !!! or the mask?
    if val_pts_option != "":
        val_x1, val_y1, val_x2, val_y2 = np.loadtxt(val_pts_option, skiprows=1, unpack=True, delimiter="\t")  # the table need to be x1, y1, x2, y2

        # Transform world coordinates into pixel coordinates (float)
        val_px1 = []; val_py1 = []; val_px2 = []; val_py2 = []
        for i in range(0, len(val_x1)):
            val_px1.append((intersection[0] - val_x1[i]) / cellsize)
            val_py1.append((val_y1[i] - intersection[1]) / cellsize)
            val_px2.append((intersection[0] - val_x2[i]) / cellsize)
            val_py2.append((val_y2[i] - intersection[1]) / cellsize)

        # Check if validation points are inside the image extent
        count = 1; store = []
        for v_px1, v_py1, v_px2, v_py2 in zip(val_px1, val_py1, val_px2, val_py2):
            if v_px1 > cols or v_px2 > cols or v_py1 > rows or v_py2 > rows:
                store.append(count)
                count = count + 1
            else:
                count = count + 1
        if store != []:
            raise Exception('The one or two points on line number', store, ' of your validation table is outside the image extent')

    return mask_data, image_information


def import_dsm(dsm1, dsm2, res_input, res_plots, gl_extent, val_pts_option):
    """
    This function is called in mode=2 (DSM mode). It imports the two input DSM (.tif) as raster. If the DSM do not have the same cell size, the DSM with the smaller
    cell size is resampled based on the DSM with the larger cell size. If the DSM do not have the same extent, both DSM are
    saved with their maximum common extent. Both DSM are saved along with their hillshades in greyscale for further processing, as well as in a lower resolution for later plots.

    Parameters
    ----------
    dsm1 : str
        Path to the first DSM
    dsm2 : str
        Path to the second DSM
    res_input: str
        Path to the input data folder
    res_plots: str
        Path to the plot folder
    gl_extent: str
        Path to the .txt file containing points delineating a processing extent (mask) if available
    val_pts_option: str
        Path to the .txt file containing validation points if available


    Returns
    -------
    mask_data : ndarray
        Two dimensional numpy array with 0 and 1. 1 is where data is available on DSM 1 and DSM 2 (the intersecting extent between both DSM) and 0 is where there is no data.

    dsm_information : list
        List containing the coordinates of image origin in the x- and y-dimension, the pixel width and height, number of rows and columns of the image as well as the projection. In order: xOrigin, pixelWidth, yOrigin, pixelHeight, rows, cols, proj

    """

    ## Message:
    print("## Importing DSMs")

    ## Check if both image paths were inserted and they have the .tif extention
    if isinstance(dsm1, str) == False or isinstance(dsm2, str) == False: print('You need to insert two DSMs')
    if dsm1.lower().endswith('.tif') == False or dsm2.lower().endswith('.tif') == False:
        raise Exception('The DSM file type must be .tif!')

    ## Open DSMs in gdal
    dsm_1 = gdal.Open(dsm1, gdalconst.GA_ReadOnly);
    dsm_2 = gdal.Open(dsm2, gdalconst.GA_ReadOnly);
    geot_dsm1 = dsm_1.GetGeoTransform()  # Geotransform: [0] =  top left x, [1] w-e pixel resolution, [3] top left y, [5] n-s pixel resolution (negative value)
    geot_dsm2 = dsm_2.GetGeoTransform()
    dsm1_band = dsm_1.GetRasterBand(1)
    dsm2_band = dsm_2.GetRasterBand(1)
    dsm1_nodata = dsm1_band.GetNoDataValue()
    dsm2_nodata = dsm2_band.GetNoDataValue()
    dsm1_proj = dsm_1.GetProjection()

    ## Find the common area between both images
    ext_dsm1 = [geot_dsm1[0], geot_dsm1[3], geot_dsm1[0] + (geot_dsm1[1] * dsm_1.RasterXSize), geot_dsm1[3] + (geot_dsm1[5] * dsm_1.RasterYSize)]
    ext_dsm2 = [geot_dsm2[0], geot_dsm2[3], geot_dsm2[0] + (geot_dsm2[1] * dsm_2.RasterXSize), geot_dsm2[3] + (geot_dsm2[5] * dsm_2.RasterYSize)]
    intersection = [max(ext_dsm1[0], ext_dsm2[0]), min(ext_dsm1[1], ext_dsm2[1]), min(ext_dsm1[2], ext_dsm2[2]), max(ext_dsm1[3], ext_dsm2[3])]
    if (intersection[2] < intersection[0]) or (intersection[1] < intersection[3]): # check if there is enough common area
        raise Exception('There is no intersection between both input DSMs!')
    if ((intersection[2] - intersection[0]) / geot_dsm1[1]) < 100 or ((intersection[1] - intersection[3]) / geot_dsm1[1]) < 100:
        raise Exception('There is too little intersection (less than 100 x 100 pixels) between both input DSMs')

    # Get larger cell size (if differences)
    cellsize = np.max([geot_dsm1[1], geot_dsm2[1]])

    ## Save the DSMs at the same resolution and extent
    gdalwarp_str_dsm1 = ''.join(('gdalwarp -overwrite -tr ', str(cellsize), ' ', str(cellsize), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ',
                                 str(intersection[2]), ' ', str(intersection[1]), ' ','-r cubic', ' ','-srcnodata', ' ', 'dsm1_nodata', ' ', dsm1, ' ', res_input + '\\D1.tif'))  # prepare str for gdalwarp module. DSM 1 saved as D1 with intersecting extent (between D1 and D2)
    os.system(gdalwarp_str_dsm1)
    gdalwarp_str_dsm2 = ''.join(('gdalwarp -overwrite -tr ', str(cellsize), ' ', str(cellsize), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ',
                                 str(intersection[2]), ' ', str(intersection[1]), ' ','-r cubic', ' ', '-srcnodata', ' ', 'dsm1_nodata', ' ', dsm2, ' ', res_input + '\\D2.tif'))  # prepare str for gdalwarp module. DSM 2 saved as D2 with cell size of DSM 1 and intersecting extent
    os.system(gdalwarp_str_dsm2)


    ## Save also the hillshade DSM
    print("Saving hillshades")
    new_name1 = res_input + '\\H1.tif'
    hillsh1 = ''.join(('gdaldem', ' ', 'hillshade', ' ',  dsm1, ' ', new_name1, ' ', '-of', ' ', 'GTiff'))
    os.system(hillsh1)
    new_name2 = res_input + '\\H2.tif'
    hillsh2 = ''.join(('gdaldem', ' ', 'hillshade', ' ',  dsm2, ' ', new_name2, ' ', '-of', ' ', 'GTiff'))
    os.system(hillsh2)

    ## Re-import DSMs
    new_dsm1 = gdal.Open(res_input + '\\D1.tif', gdalconst.GA_ReadOnly)
    new_dsm2 = gdal.Open(res_input + '\\D2.tif', gdalconst.GA_ReadOnly)
    new_dsm1_arr = new_dsm1.ReadAsArray().astype(np.float)
    new_dsm2_arr = new_dsm2.ReadAsArray().astype(np.float)
    rows, cols = new_dsm1_arr.shape[:2]  # new_dsm1_arr and new_dsm2_arr have the same shape

    # Get information of the new DSMs
    dsm_information = (intersection[0], cellsize, intersection[1], -cellsize, rows, cols, dsm1_proj)

    # Resize the above DSMs and hillshade that will be used for plotting in a later function
    print("Exporting hillshades in lower resolution for plotting")
    resizing = 2  # to 2m resolution
    gdalwarp_H1 = ''.join(('gdalwarp -overwrite -tr ', str(resizing), ' ', str(resizing), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ',str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', res_input + '\\H1.tif', ' ', res_plots + '\\H1_resize.tif'))
    os.system(gdalwarp_H1)
    gdalwarp_H2 = ''.join(('gdalwarp -overwrite -tr ', str(resizing), ' ', str(resizing), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ',str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', res_input + '\\H2.tif', ' ', res_plots + '\\H2_resize.tif'))
    os.system(gdalwarp_H2)

    ## Create mask in which the pre-filter, matching and postfilter will occur
    # If user added a glacier_extent file use this one, otherwise use the common glacier extent calculated above.
    if gl_extent != "":
        ext_x, ext_y = np.loadtxt(gl_extent, skiprows=1, unpack=True, delimiter="\t")

        # Transform world coordinates into pixel coordinates (float)
        ext_px = []; ext_py = []
        for i in range(0, len(ext_x)):
            ext_py.append((intersection[1] - ext_y[i]) / cellsize)
            ext_px.append((ext_x[i] - intersection[0]) / cellsize)

        # Check if extent points are inside the DSM extent
        count = 1; store = []
        for v_px, v_py in zip(ext_px, ext_py):
            if v_px > cols or v_py > rows:
                store.append(count)
                count = count + 1
            else:
                count = count + 1
        if store != []:
            raise Exception('The point on line number', store,' of your glacier extent table is outside the image extent')

        # Tranform the points in polygon
        polygon = list(zip(ext_px, ext_py))
        imgaage = Image.new('L', (cols, rows), 0)
        ImageDraw.Draw(imgaage).polygon(polygon, outline=1, fill=1)
        mask_data = np.array(imgaage)
    else:
        # Create intersecting mask (mask = 1, otherwise 0) between both dsms
        a = new_dsm1_arr != dsm1_nodata
        a2 = new_dsm2_arr != dsm2_nodata
        mask_data = np.where((a == 1) & (a2 == 1), 1, 0)

    # Check whether the validation points (if there are) are inside the image extent !!!
    if val_pts_option != "":
        val_x1, val_y1, val_x2, val_y2 = np.loadtxt(val_pts_option, skiprows=1, unpack=True, delimiter="\t")  # the table needs to be x1,y1, x2,y2

        # Transform world coordinates into pixel coordinates (in float)
        val_px1 = []; val_py1 = []; val_px2 = []; val_py2 = []
        for i in range(0, len(val_x1)):
            val_px1.append((intersection[0] - val_x1[i]) / cellsize)
            val_py1.append((val_y1[i] - intersection[1]) / cellsize)
            val_px2.append((intersection[0] - val_x2[i]) / cellsize)
            val_py2.append((val_y2[i] - intersection[1]) / cellsize)

        # Check if validation points are inside the image extent
        count = 1; store = []
        for v_px1, v_py1, v_px2, v_py2 in zip(val_px1, val_py1, val_px2, val_py2):
            if v_px1 > cols or v_px2 > cols or v_py1 > rows or v_py2 > rows:
                store.append(count)
                count = count + 1
            else:
                count = count + 1
        if store != []:
            raise Exception('The one or two points on line number', store, ' of your validation table is outside the DSM extent')


    return mask_data, dsm_information


def import_datasets(image1, image2, dsm1, dsm2, res_input, res_plots, gl_extent, val_pts_option):
    """
    This function is called in mode=2 (DSM mode). It imports the two input DSM (.tif) as raster. If the DSM do not have the same cell size, the DSM with the smaller
    cell size is resampled based on the DSM with the larger cell size. If the DSM do not have the same extent, both DSM are
    saved with their maximum common extent. Both DSM are saved along with their hillshades in greyscale for further processing, as well as in a lower resolution for later plots.

    Parameters
    ----------
    image1 : str
        Path to the first image
    image2 : str
        Path to the second image
    dsm1 : str
        Path to the first DSM
    dsm2 : str
        Path to the second DSM
    res_input: str
        Path to the input data folder
    res_plots: str
        Path to the plot folder
    gl_extent: str
        Path to the .txt file containing points delineating a processing extent (mask) if available
    val_pts_option: str
        Path to the .txt file containing validation points if available


    Returns
    -------
    mask_data : ndarray
        Two dimensional numpy array with 0 and 1. 1 are the pixels that will be used in later analysis.
    dataset_information : list
        List containing the coordinates of dataset origin in the x- and y-dimension, the pixel width and height, number of rows and columns of the image as well as the projection. In order: xOrigin, pixelWidth, yOrigin, pixelHeight, rows, cols, proj

    """

    ## Message:
    print("## Import images and DSMs")

    ## Check if both image paths were inserted and they have the .tif extention
    if isinstance(image1, str) == False or isinstance(image2, str) == False: raise Exception('You need to insert two images')
    if image1.lower().endswith('.tif') == False or image2.lower().endswith('.tif') == False: raise Exception('The images file type must be tif!')  # later check for gtif, gtiff, jp2, jpeg2000, or geojp2.  --> im2.lower().endswith(('.png', '.jpg', '.jpeg'))

    ## Check if both DSM paths were inserted and they have the .tif extention
    if isinstance(dsm1, str) == False or isinstance(dsm2, str) == False: print('You need to insert two DSMs')
    if dsm1.lower().endswith('.tif') == False or dsm2.lower().endswith('.tif') == False: raise Exception('The DSMs file type must be tif!')

    ## Open images in gdal
    img1 = gdal.Open(image1, gdalconst.GA_ReadOnly)
    img2 = gdal.Open(image2, gdalconst.GA_ReadOnly)
    geot_img1 = img1.GetGeoTransform()  # Geotransform: [0] =  top left x, [1] w-e pixel resolution, [3] top left y, [5] n-s pixel resolution (negative value)
    geot_img2 = img2.GetGeoTransform()
    img1_proj = img1.GetProjection()

    ## Find the common area between both images
    ext_img1 = [geot_img1[0], geot_img1[3], geot_img1[0] + (geot_img1[1] * img1.RasterXSize), geot_img1[3] + (geot_img1[5] * img1.RasterYSize)]
    ext_img2 = [geot_img2[0], geot_img2[3], geot_img2[0] + (geot_img2[1] * img2.RasterXSize), geot_img2[3] + (geot_img2[5] * img2.RasterYSize)]
    intersection1 = [max(ext_img1[0], ext_img2[0]), min(ext_img1[1], ext_img2[1]), min(ext_img1[2], ext_img2[2]), max(ext_img1[3], ext_img2[3])]
    if (intersection1[2] < intersection1[0]) or (intersection1[1] < intersection1[3]):raise Exception('There is no intersection between both input images')
    if ((intersection1[2] - intersection1[0]) / geot_img1[1]) < 100 or ((intersection1[1] - intersection1[3]) / geot_img1[1]) < 100:
        raise Exception('There is too little intersection (less than 100 x 100 pixels) between both input images')


    ## Open DSMs in gdal
    dsm_1 = gdal.Open(dsm1, gdalconst.GA_ReadOnly);
    dsm_2 = gdal.Open(dsm2, gdalconst.GA_ReadOnly);
    geot_dsm1 = dsm_1.GetGeoTransform();  # Geotransform: [0] =  top left x, [1] w-e pixel resolution, [3] top left y, [5] n-s pixel resolution (negative value)
    geot_dsm2 = dsm_2.GetGeoTransform();
    dsm1_band = dsm_1.GetRasterBand(1)
    dsm2_band = dsm_2.GetRasterBand(1)
    dsm1_nodata = dsm1_band.GetNoDataValue()
    dsm2_nodata = dsm2_band.GetNoDataValue()

    ## Find the common area between both DSMs
    ext_dsm1 = [geot_dsm1[0], geot_dsm1[3], geot_dsm1[0] + (geot_dsm1[1] * dsm_1.RasterXSize),geot_dsm1[3] + (geot_dsm1[5] * dsm_1.RasterYSize)]
    ext_dsm2 = [geot_dsm2[0], geot_dsm2[3], geot_dsm2[0] + (geot_dsm2[1] * dsm_2.RasterXSize),geot_dsm2[3] + (geot_dsm2[5] * dsm_2.RasterYSize)]
    intersection2 = [max(ext_dsm1[0], ext_dsm2[0]), min(ext_dsm1[1], ext_dsm2[1]), min(ext_dsm1[2], ext_dsm2[2]), max(ext_dsm1[3], ext_dsm2[3])]
    if (intersection2[2] < intersection2[0]) or (intersection2[1] < intersection2[3]):  # check if there is enough common area
        raise Exception('There is no intersection between both input DSMs!')
    if ((intersection2[2] - intersection2[0]) / geot_dsm1[1]) < 100 or ((intersection2[1] - intersection2[3]) / geot_dsm1[1]) < 100:
        raise Exception('There is too little intersection (less than 100 x 100 pixels) between both input DSMs')

    # Get common intersection between images and DSMs
    intersection = [max(intersection1[0], intersection2[0]), min(intersection1[1], intersection2[1]),min(intersection1[2], intersection2[2]), max(intersection1[3], intersection2[3])]

    # Get larger cell size
    cellsize = np.max([geot_img1[1], geot_img2[1], geot_dsm1[1], geot_dsm2[1]])

   ## Save datasets to the same resolution and extent
    # Images
    gdalwarp_str_img1 = ''.join(('gdalwarp -overwrite -tr ', str(cellsize), ' ', str(cellsize), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ', str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', image1, ' ', res_input + '\\i1.tif'))  # prepare str for gdalwarp module. Image 1 saved as I1 with intersecting extent (between I1 and I2
    os.system(gdalwarp_str_img1)

    gdalwarp_str_img2 = ''.join(('gdalwarp -overwrite -tr ', str(cellsize), ' ', str(cellsize), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ', str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', image2, ' ', res_input + '\\i2.tif'))  # prepare str for gdalwarp module. Image 2 saved as I2 with cell size of image 1 and intersecting extent
    os.system(gdalwarp_str_img2)

    # DSMs
    gdalwarp_str_dsm1 = ''.join(('gdalwarp -overwrite -tr ', str(cellsize), ' ', str(cellsize), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ', str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', '-srcnodata', ' ', 'dsm1_nodata', ' ', dsm1, ' ', res_input + '\\D1.tif'))  # prepare str for gdalwarp module. DSM 1 saved as D1 with intersecting extent (between D1 and D2)
    os.system(gdalwarp_str_dsm1)

    gdalwarp_str_dsm2 = ''.join(('gdalwarp -overwrite -tr ', str(cellsize), ' ', str(cellsize), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ', str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', '-srcnodata', ' ', 'dsm1_nodata', ' ', dsm2, ' ', res_input + '\\D2.tif'))  # prepare str for gdalwarp module. DSM 2 saved as D2 with cell size of DSM 1 and intersecting extent
    os.system(gdalwarp_str_dsm2)

    ## Hillshades
    new_name1 = res_input + '\\H1.tif'
    hillsh1 = ''.join(('gdaldem', ' ', 'hillshade', ' ', dsm1, ' ', new_name1, ' ', '-of', ' ', 'GTiff'))
    os.system(hillsh1)
    new_name2 = res_input + '\\H2.tif'
    hillsh2 = ''.join(('gdaldem', ' ', 'hillshade', ' ', dsm2, ' ', new_name2, ' ', '-of', ' ', 'GTiff'))
    os.system(hillsh2)

    ## Re-import datasets
    # Re-import images
    new_image1 = skimage.data.load(os.path.abspath(res_input + '\\i1.tif'))
    new_image2 = skimage.data.load(os.path.abspath(res_input + '\\i2.tif'))
    new_img_geotransf = (intersection[0], cellsize, 0.0, intersection[1], 0.0, -cellsize)

    # Re-import DSMs
    new_dsm1 = gdal.Open(res_input + '\\D1.tif', gdalconst.GA_ReadOnly)
    new_dsm2 = gdal.Open(res_input + '\\D2.tif', gdalconst.GA_ReadOnly)
    new_dsm1_arr = new_dsm1.ReadAsArray().astype(np.float)
    new_dsm2_arr = new_dsm2.ReadAsArray().astype(np.float)

    rows, cols = new_image1.shape[:2]  # new_image1 and new_image2, new_dsm1 and new_dsm2 have the same shape
    dataset_information = (intersection[0], cellsize, intersection[1], -cellsize, rows, cols, img1_proj)  # xOrigin, pixelWidth, yOrigin, pixelHeight, rows, cols, proj

    # Store images in grayscale
    if len(new_image1.shape) == 2:  # Check whether the uploaded images are already grayscale
        image_gray1 = new_image1
    elif new_image1.shape[2] == 3:  # Check whether the image have 3 channels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # to ignore: UserWarning: Possible precision loss when converting from float64 to uint8
            image_gray1 = skimage.img_as_ubyte(rgb2gray(new_image1))
    elif new_image1.shape[2] == 4:  # Check whether the image have 4 channels
        image_rgb = rgba2rgb(new_image1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # to ignore: UserWarning: Possible precision loss when converting from float64 to uint8
            image_gray1 = skimage.img_as_ubyte(rgb2gray(image_rgb))  # There is no direct function rgba2gray (yet)
    else:
        raise Exception('Image 1 is not grayscale, RGB or RGBA')

    if len(new_image2.shape) == 2:
        image_gray2 = new_image2
    elif new_image2.shape[2] == 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image_gray2 = skimage.img_as_ubyte(rgb2gray(new_image2))
    elif new_image2.shape[2] == 4:
        image_rgb = rgba2rgb(new_image2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image_gray2 = skimage.img_as_ubyte(rgb2gray(image_rgb))
    else:
        raise Exception('Image 2 not grayscale, RGB or RGBA')


    ## Save the new images (the ones that will be used in further processing) in the temporary file
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(res_input + '\\I1.tif', cols, rows, 1, gdal.GDT_Byte, )
    dataset.SetGeoTransform(new_img_geotransf)
    dataset.SetProjection(img1_proj)
    dataset.GetRasterBand(1).WriteArray(image_gray1)
    dataset.FlushCache()  # Write to disk.

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(res_input + '\\I2.tif', cols, rows, 1, gdal.GDT_Byte, )
    dataset.SetGeoTransform(new_img_geotransf)
    dataset.SetProjection(img1_proj)
    dataset.GetRasterBand(1).WriteArray(image_gray2)
    dataset.FlushCache()


    ## Create mask in which the pre-filter, matching and postfilter will occur
    if gl_extent != "":  # if user added a glacier_extent file use this one, otherwise use the common glacier extent calculated above.
        ext_x, ext_y = np.loadtxt(gl_extent, skiprows=1, unpack=True, delimiter="\t")

        # Transform world coordinates into pixel coordinates (float)
        ext_px = []; ext_py = []
        for i in range(0, len(ext_x)):
            ext_py.append((intersection[1] - ext_y[i]) / cellsize)
            ext_px.append((ext_x[i] - intersection[0]) / cellsize)

        # Check if extent points are inside the image extent
        count = 1; store = []
        for v_px, v_py in zip(ext_px, ext_py):
            if v_px > cols or v_py > rows:
                store.append(count)
                count = count + 1
            else:
                count = count + 1
        if store != []:
            raise Exception('The point on line number', store, ' of your glacier extent table is outside the image extent')

        # Transform the points in polygon
        polygon = list(zip(ext_px, ext_py))
        imgaage = Image.new('L', (cols, rows), 0)
        ImageDraw.Draw(imgaage).polygon(polygon, outline=1, fill=1)
        mask_data = np.array(imgaage)
    else:
        ## Create intersecting mask (mask = 1, otherwise 0) between both images
        a = image_gray1 != 255  # takes out white background
        b = image_gray1 != 0  # takes out white background
        a2 = image_gray2 != 255
        b2 = image_gray2 != 0

        # Create intersecting mask (mask = 1, otherwise 0) between both dsms
        c = new_dsm1_arr != dsm1_nodata
        c2 = new_dsm2_arr != dsm2_nodata

        # Intersect both masks
        mask_data = np.where((a == 1) & (b == 1) & (a2 == 1) & (b2 == 1) & (c == 1) & (c2 == 1), 1, 0)

    # end of gl_extent loop


    # Resize the above images that will be used as ploting in a later function
    print("Exporting datasets in lower resolution for plotting")
    resizing = (2 / cellsize) * cellsize  # to 2 m resolution

    gdalwarp_I1 = ''.join(('gdalwarp -overwrite -tr ', str(resizing), ' ', str(resizing), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ', str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', res_input + '\\I1.tif', ' ', res_plots + '\\I1_resize.tif'))
    os.system(gdalwarp_I1)
    gdalwarp_I2 = ''.join(('gdalwarp -overwrite -tr ', str(resizing), ' ', str(resizing), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ', str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', res_input + '\\I2.tif', ' ', res_plots + '\\I2_resize.tif'))
    os.system(gdalwarp_I2)

    gdalwarp_H1 = ''.join(('gdalwarp -overwrite -tr ', str(resizing), ' ', str(resizing), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ',str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', res_input + '\\H1.tif', ' ', res_plots + '\\H1_resize.tif'))
    os.system(gdalwarp_H1)
    gdalwarp_H2 = ''.join(('gdalwarp -overwrite -tr ', str(resizing), ' ', str(resizing), ' ', '-te', ' ', str(intersection[0]), ' ', str(intersection[3]), ' ',str(intersection[2]), ' ', str(intersection[1]), ' ', '-r cubic', ' ', res_input + '\\H2.tif', ' ', res_plots + '\\H2_resize.tif'))
    os.system(gdalwarp_H2)

    # Check weather the validation points (if there are) are inside the dataset extent !!! or the mask?
    if val_pts_option != "":
        val_x1, val_y1, val_x2, val_y2 = np.loadtxt(val_pts_option, skiprows=1, unpack=True, delimiter="\t")  # table needs to be x1,y1, x2,y2

        # Transform world coordinates into pixel coordinates (float)
        val_px1 = []; val_py1 = []; val_px2 = []; val_py2 = []
        for i in range(0, len(val_x1)):
            val_px1.append((intersection[0] - val_x1[i]) / cellsize)
            val_py1.append((val_y1[i] - intersection[1]) / cellsize)
            val_px2.append((intersection[0] - val_x2[i]) / cellsize)
            val_py2.append((val_y2[i] - intersection[1]) / cellsize)

        # Check if validation points are inside the image extent
        count = 1
        store = []
        for v_px1, v_py1, v_px2, v_py2 in zip(val_px1, val_py1, val_px2, val_py2):
            if v_px1 > cols or v_px2 > cols or v_py1 > rows or v_py2 > rows:
                store.append(count)
                count = count + 1
            else:
                count = count + 1
        if store != []:
            raise Exception('The one or two points on line number', store, ' of your validation table is outside the image extent')


    return mask_data, dataset_information


def prefilter_img(image_filter_list, res_input, mask_data):
    """
    This function imports the input images and applies the filters defined from the user in the params.py file  ([F1, F2,...]).

    Parameters
    ----------

    image_filter_list : list
        List defined by the user in the params.py file that has indicates which filter will be applied on the images
    res_input : str
        Path to input dataset folder
    mask_data : ndarray
        Array of 1 and 0. The values 1 represent the mask intersecting image 1 and image 2


    Returns
    -------
    None: The input images are filtered and saved in the input folder.

    """

    ## Message:
    print("## Pre-filtering images")

    ## Import images from (temporary) folder
    files_img = os.listdir(res_input)
    files_img_list = []
    for file in files_img:
        if file.endswith("tif"):
            pass
        else: raise Exception('The imported filed need to be .tif')

        if 'I' in file: files_img_list.append(os.path.join(res_input, file))

    counter = [1, 2, 4, 8]
    ## Apply the selected filters
    for img in files_img_list:
        for element in image_filter_list:
            image = cv2.imread(os.path.abspath(img), 0)
            if len(image.shape) != 2: raise Exception('Your image ', img, ' should be grayscale')
            count = 0
            if 'F1' in element:
                count = count + counter[0]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # to ignore: UserWarning: Possible precision loss when converting from float64 to uint16
                    image = skimage.filters.median(image, selem=None, out=None, mask=mask_data, shift_x=False, shift_y=False)
                    image[np.where(mask_data == 0)] = 255  # skimage.filters.median filter returns a black (0) background. Reconvert in white (255). It saves the image with the mask shape in both options
            if 'F2' in element:
                count = count + counter[1]
                image = skimage.exposure.equalize_hist(image, nbins=256, mask=mask_data)  # alternative: image = cv2.equalizeHist(image)
            if 'F3' in element:
                count = count + counter[2]
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # alternative: skimage.exposure.equalize_adapthist(image)
                image = clahe.apply(image)

            # Define name + save
            if count == 1: filt_name = 'F1.tif'
            if count == 2: filt_name = 'F2.tif'
            if count == 3: filt_name = 'F1F2.tif'
            if count == 4: filt_name = 'F3.tif'
            if count == 5: filt_name = 'F1F3.tif'
            if count == 6: filt_name = 'F2F3.tif'

            img_filename = os.path.splitext(os.path.basename(img))[0]
            new_name_img = res_input + "\\" + img_filename + filt_name
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # to ignore: UserWarning: Possible precision loss when converting from float64 to uint8
                skimage.io.imsave(new_name_img, image, plugin='pil')

    return


def prefilter_dsm(dsm_filter_list, res_input):
    """
    This function imports the input DSMs and applies the filters defined from the user in the params.py file  ([F1, F2,...]).

    Parameters
    ----------

    dsm_filter_list : list
        List defined by the user in the params.py file that has indicates which filter will be applied on the DSMs
    res_input : str
        Path to input dataset folder


    Returns
    -------
    None: The input DSMs are filtered and saved in the input folder.

    """

    ## Message:
    print("## Pre-filtering DSMs")

    ## Import dsms from (temporary) folder
    files_dsm = os.listdir(res_input)
    files_dsm_list = []
    for file in files_dsm:
        if file.endswith("tif"): pass
        else: raise Exception('The imported filed need to be .tif')
        if 'D' in file: files_dsm_list.append(os.path.join(res_input, file))

    counter = [1, 2, 4, 8]
    ## Apply the selected filters
    for dsms in files_dsm_list:
        for element in dsm_filter_list:
            dsm = gdal.Open(dsms, gdalconst.GA_ReadOnly)
            dsm_proj = dsm.GetProjection()
            dsm_tr = dsm.GetGeoTransform()
            dsm_arr = dsm.ReadAsArray()
            dsm_arr[dsm_arr == -9999] = np.nan
            count = 0
            if 'F1' in element:
                count = count + counter[0]
                dsm_arr = cv2.medianBlur(dsm_arr, 3)
            if 'F4' in element:
                count = count + counter[1]
                dsm_arr = cv2.bilateralFilter(dsm_arr, 5, 10, 10)

            # Define name + save
            if count == 1: filt_name = 'F1.tif'
            if count == 2: filt_name = 'F4.tif'
            if count == 3: filt_name = 'F1F4.tif'

            dsm_filename = os.path.splitext(os.path.basename(dsms))[0]
            new_name_dsm = res_input + "\\" + dsm_filename + filt_name

            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(new_name_dsm, dsm.RasterXSize, dsm.RasterYSize, 1, gdal.GDT_Float32, )
            dataset.SetGeoTransform(dsm_tr)
            dataset.SetProjection(dsm_proj)
            dataset.GetRasterBand(1).WriteArray(dsm_arr)
            dataset.FlushCache()  # Write to disk.
            dataset = None

    return


def prefilter_hil(hil_filter_list, res_input, mask_data):
    """
    This function imports the input hillshades and applies the filters defined from the user in the params.py file  ([F1, F2,...]).

    Parameters
    ----------

    dsm_filter_list : list
        List defined by the user in the params.py file that has indicates which filter will be applied on the hillshades
    res_input : str
        Path to input images folder
    mask_data : ndarray
        Array of 1 and 0. The values 1 represent the mask intersecting hillshade 1 and hillshade 2

    Returns
    -------
    None: The input hillshade are filtered and saved in the input folder.

    """

    ## Message:
    print("## Pre-filtering hillshades")

    ## Import hillshades from input data folder
    files_hil = os.listdir(res_input);
    files_hil_list = [];
    for file in files_hil:
        if file.endswith("tif"): pass
        else: raise Exception('The imported filed need to be .tif')
        if 'H' in file: files_hil_list.append(os.path.join(res_input, file))

    counter = [1, 2, 4, 8]
    ## Apply the selected filters
    for hil in files_hil_list:
        for element in hil_filter_list:
            hill = skimage.data.load(os.path.abspath(hil))
            if len(hill.shape) != 2: raise Exception('Your image ', hil, ' should be grayscale')
            count = 0
            if 'F1' in element:
                count = count + counter[0]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # to ignore: UserWarning: Possible precision loss when converting from float64 to uint16
                    hill = skimage.filters.median(hill, selem=None, out=None, mask=mask_data, shift_x=False, shift_y=False)
            if 'F5' in element:
                count = count + counter[1]
                hill = skimage.filters.sobel(hill, mask=mask_data)  # alternative: skimage.feature.canny

            # Define name + save
            if count == 1: filt_name = 'F1.tif'
            if count == 2: filt_name = 'F5.tif'
            if count == 3: filt_name = 'F1F5.tif'

            hill_filename = os.path.splitext(os.path.basename(hil))[0]
            new_name_hil = res_input + "\\" + hill_filename + filt_name
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # to ignore: UserWarning: Possible precision loss when converting from float64 to uint8
                skimage.io.imsave(new_name_hil, hill, plugin='pil')

    return


def matching(res_input, res_matching, method_list, mask_data, step_grid, pix_dev, nbr_win, data_info):
    """
    This function should match two images with different matching methods.

   Parameters
   ----------

   res_input : str
       Path to input dataset folder
   res_matching : str
       Path to the (temporary) matching folder, where the matching tables are stored
   method_list : list
       List defined by the user in the params.py file that has indicates which methods will be applied on the datasets.
   mask_data : ndarray
        Array of 1 and 0. The values 1 represent the mask intersecting dataset 1 and dataset 2
   step_grid: int
        Spacing chosen by the user in the params.py file that defines when a matching measurement took place. The number is in pixel.
   pix_dev : int
        The estimated displacement between both epochs in pixel. This parameter is defined in the params.py file
   nbr_win : int
        Number of windows that will be tested. This parameter is defined in the params.py file
   data_info : list
        List containing the coordinates of dataset origin in the x- and y-dimension, the pixel width and height, number of rows and columns of the image as well as the projection. In order: xOrigin, pixelWidth, yOrigin, pixelHeight, rows, cols, proj


   Returns
   -------
    [pv_m, pu_m] : list
        List of pixel coordinates (rows, columns) of all points 'p' that were matched

    """

    ## Message:
    print("## Matching")

    ## Import input data from the(temporary) folder
    files_input = os.listdir(res_input)
    files1 = []; files2 = []
    for file in files_input:
        if 'I1' in file or 'D1' in file or 'H1' in file:
            files1.append(os.path.join(res_input, file))
        if 'I2' in file or 'D2' in file or 'H2' in file:
            files2.append(os.path.join(res_input, file))
    if np.size(files1) != np.size(files2):
        raise Exception('The number of input data 1 and 2 (data pairs) in the temporary folder must be equal!')

    files1.sort(); files2.sort()

    ## Get information from input images
    xOrigin, pixelWidth, yOrigin, pixelHeight, rows, cols, proj = data_info

    ## Points to be matched
    pv, pu = np.meshgrid(np.arange(0, rows, step_grid), np.arange(0, cols, step_grid), indexing='ij')  # pv=rows, pu=cols
    pv_1D = pv.flatten()
    pu_1D = pu.flatten()

    # Only select points within the mask
    pv_m = []; pu_m = []
    for i in range(len(pv_1D)):
        if mask_data[pv_1D[i], pu_1D[i]] == 1:
            pv_m.append(pv_1D[i])
            pu_m.append(pu_1D[i])

    ## Create window sizes
    wsize_min, wsize_max = (pix_dev * 6), (pix_dev * 17)  # !!!advanced_param: here the size of the minimum and maximum window size can be changed. Default values 6 and 17 (Bickel et al. 2018, Quantitative Assessment of Digital Image Correlation Methods to Detect and Monitor Surface Displacements of Large Slope Instabilities).

    if nbr_win == 1:
        match_win_size = np.array([int((wsize_min + wsize_max)/2)])
        if match_win_size[0] % 2 == 0: match_win_size[0] = match_win_size[0]+1  # only odd numbers
    else:
        match_win_size = np.linspace(wsize_min, wsize_max, num=nbr_win, endpoint=True).astype(int)
        match_win_size = np.sort([x+1 for x in match_win_size if x % 2 == 0] + [x for x in match_win_size if x % 2 != 0])  # only odd numbers


    ## --- START MATCHING for all image pairs
    print("Estimating remaining time... this will take a while")

    # Matching method 1 (Normalized Cross Correlation (NCC))
    if 'M1' in method_list:
        print("Matching method 1")
        start_time_M1 = time.time()

        count_pairs = 1
        start_time_pair = time.time()
        for f1, f2 in zip(files1, files2):
            file_names = os.path.splitext(os.path.basename(f1))[0]
            file_names = file_names[:1] + file_names[2:]  # take out second character (no need of image numbers anymore)

            # Open image pair
            img1 = gdal.Open(f1, gdalconst.GA_ReadOnly)
            img2 = gdal.Open(f2, gdalconst.GA_ReadOnly)
            img1_array = img1.ReadAsArray().astype(np.float)
            img2_array = img2.ReadAsArray().astype(np.float)
            img1_array = img1_array
            img2_array = img2_array

            count_win = 1  # counter for number of windows
            start_time_win = time.time()
            for template_size in match_win_size:  # loop over each template size
                print("Data pair: ", count_pairs, "/", np.size(files1), "Window: ", count_win, "/", np.size(match_win_size))
                search_window = template_size + (pix_dev*2)  # *2 because the displacement can take place in all directions
                if search_window % 2 == 0: search_window = search_window + 1  # only odd numbers are possible to get a central pixel

                htemp_size = int(template_size/2)
                hsearch_win = int(search_window/2)
                proc_size = template_size + search_window -1

                # Padding arrays
                img1_pad = np.pad(img1_array, [(hsearch_win,), (hsearch_win,)], 'constant', constant_values=0.0)
                img2_pad = np.pad(img2_array, [(hsearch_win,), (hsearch_win,)], 'constant', constant_values=0.0)

                # Correct pv_m and pu_m values based on padding
                pu_m = [x + hsearch_win for x in pu_m]
                pv_m = [x + hsearch_win for x in pv_m]

                peakCorr = np.empty(len(pv_m)) * np.nan
                meanAbsCorr = np.empty(len(pv_m)) * np.nan
                d_x2 = np.empty(len(pv_m)) * np.nan
                d_y2 = np.empty(len(pv_m)) * np.nan
                for i in range(len(pv_m)): # Make moving template on selected points 'p'
                    template = img1_pad[pv_m[i]-htemp_size: pv_m[i]+(htemp_size+1), pu_m[i]-htemp_size: pu_m[i]+(htemp_size+1)]  # +1 to put the p in the center
                    search_win = img2_pad[pv_m[i]-hsearch_win : pv_m[i]+(hsearch_win+1), pu_m[i]-hsearch_win: pu_m[i]+(hsearch_win+1)]

                    pad = int((search_win.shape[0] - template.shape[0]) / 2)

                    meanT = np.nanmean(template)
                    sigmaT = np.sqrt(np.sum(((template-meanT)**2)))

                    if sigmaT != 0:
                        fft_template = np.fft.fft2(np.rot90(template, 2), s=(proc_size, proc_size))
                        fft_search = np.fft.fft2(search_win, s=(proc_size, proc_size))
                        result = np.fft.ifft2(fft_template * fft_search)

                        # Calculate local sum of search_win
                        zp = np.zeros((np.shape(search_win)[0], np.shape(template)[0]))
                        t_cum1 = np.cumsum(np.concatenate((zp, search_win, zp), axis=1), 1)
                        zp = np.zeros((np.shape(template)[0], np.shape(t_cum1)[1]))
                        conc = np.concatenate((zp, t_cum1, zp), axis=0)

                        b1 = np.shape(template)[1]; b2 = np.shape(t_cum1)[1]; b3 = b2 - b1
                        tot = (conc[:, b1:b2]) - (conc[:, 0:b3])
                        t_cum2 = np.cumsum(tot, axis=0)
                        lsum = (t_cum2[b1:b2, :]) - (t_cum2[0:b3, :])
                        lsum_search = lsum[0:len(lsum) - 1, 0:len(lsum) - 1]

                        # Calculate local sum of search_win*search_win
                        search_win2 = search_win*search_win

                        zp = np.zeros((np.shape(search_win2)[0], np.shape(template)[0]))
                        t_cum1 = np.cumsum(np.concatenate((zp, search_win2, zp), axis=1), 1)
                        zp = np.zeros((np.shape(template)[0], np.shape(t_cum1)[1]))
                        conc = np.concatenate((zp, t_cum1, zp), axis=0)

                        b1 = np.shape(template)[1]; b2 = np.shape(t_cum1)[1]; b3 = b2 - b1
                        tot = (conc[:, b1:b2]) - (conc[:, 0:b3])
                        t_cum2 = np.cumsum(tot, axis=0)
                        lsum = (t_cum2[b1:b2, :]) - (t_cum2[0:b3, :])
                        lsum_search2 = lsum[0:len(lsum) - 1, 0:len(lsum) - 1]

                        sigmaB = np.sqrt(lsum_search2 - (lsum_search ** 2) / np.size(template))

                        with np.errstate(divide='ignore', invalid='ignore'):
                            result = np.real((result - lsum_search * meanT) / (sigmaT * sigmaB))

                        result[np.isnan(result)] = 0

                    else:
                        result = np.zeros((proc_size, proc_size))

                    # Cropping results not affected by edges
                    result_nopad = result[np.int((proc_size/2)-pad): np.int((proc_size/2)+(pad+1)), np.int((proc_size/2)-pad): np.int((proc_size/2)+(pad+1))]

                    # It can be that there are inf and -inf. These points are set to 0
                    id_inf_p = np.where(np.isposinf(result_nopad))
                    result_nopad[id_inf_p] = 0
                    id_inf_n = np.where(np.isneginf(result_nopad))
                    result_nopad[id_inf_n] = 0
                    x_res = np.array(range(-int(len(result_nopad)/2), int(len(result_nopad)/2)+1))
                    y_res = range(-int(len(result_nopad)/2), int(len(result_nopad)/2)+1)

                    # Get coordinates of max value and max value
                    y_m, x_m = np.unravel_index(np.argmax(result_nopad), result_nopad.shape)  # row, col as int.
                    max_val = np.max(result_nopad)

                    # --- Sub-pixel accuracy
                    # Check whether the peak is on an edge
                    edgedist = np.min(np.abs([0 - y_m, 0 - x_m,  y_m - np.shape(result_nopad)[0], x_m - np.shape(result_nopad)[0]]))

                    if edgedist == 0:  # don't take this point into account
                        subpix_deviation_y = np.nan
                        subpix_deviation_x = np.nan

                    elif edgedist == 1:
                        win_subpix = 3 ; hwin_subpix = int(win_subpix / 2)  # 3x3 window

                        # Take part of the correlation around the max peak
                        result_win = result_nopad[y_m - hwin_subpix: y_m + (hwin_subpix + 1), x_m - hwin_subpix: x_m + (hwin_subpix + 1)]
                        y_sub, x_sub = np.meshgrid(y_res[y_m - hwin_subpix: y_m + (hwin_subpix + 1)], x_res[x_m - hwin_subpix: x_m + (hwin_subpix + 1)], indexing='ij')

                        result_win_flat = np.ndarray.flatten(result_win)
                        middle = int((len(result_win_flat) - 1) / 2)
                        np.delete(result_win_flat, middle)
                        result_win_mean = np.mean(result_win_flat)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            result_win = result_win - result_win_mean
                            result_win = result_win / np.sum(result_win)

                            subpix_deviation_y = np.sum(y_sub * result_win)
                            subpix_deviation_x = np.sum(x_sub * result_win)

                    else:
                        win_subpix = 5; hwin_subpix = int(win_subpix/2)  # !!!advanced_param: Here the size of the window from which subpixel displacement is derived can be changed. default = 5x5 window

                        # Take part of the correlation around the max peak
                        result_win = result_nopad[y_m-hwin_subpix : y_m+(hwin_subpix+1), x_m-hwin_subpix : x_m+(hwin_subpix+1)]
                        y_sub, x_sub = np.meshgrid(y_res[y_m-hwin_subpix : y_m+(hwin_subpix+1)], x_res[x_m-hwin_subpix : x_m+(hwin_subpix+1)], indexing='ij')

                        result_win_flat = np.ndarray.flatten(result_win)
                        middle = int((len(result_win_flat) - 1)/2)
                        np.delete(result_win_flat, middle)
                        result_win_mean = np.mean(result_win_flat)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            result_win = result_win - result_win_mean
                            result_win[result_win < 0] = 0
                            result_win = result_win / np.sum(result_win)

                        subpix_deviation_y = np.sum(y_sub * result_win)
                        subpix_deviation_x = np.sum(x_sub * result_win)

                    # Save values for point 'p'
                    if np.isnan(subpix_deviation_y):
                        peakCorr[i] = np.nan
                        meanAbsCorr[i] = np.nan
                        d_y2[i] = subpix_deviation_y  # rows
                        d_x2[i] = subpix_deviation_x  # cols
                    else:
                        peakCorr[i] = max_val
                        meanAbsCorr[i] = np.nanmean(np.abs(result_nopad))
                        d_y2[i] = subpix_deviation_y
                        d_x2[i] = subpix_deviation_x

                # end of moving template on selected points 'p' loop

                try:
                    SNratio = peakCorr / meanAbsCorr
                except RuntimeWarning:
                    SNratio = np.nan

                # Reset correct pv_m and pu_m values without padding
                pu_m = [x - hsearch_win for x in pu_m]
                pv_m = [x - hsearch_win for x in pv_m]

                ##--- SAVE RESULTS FROM MATCHING ---
                table_base_name = res_matching + '\\' + file_names + 'M1W' + str(template_size).zfill(4)  + '.txt'
                table_array = np.stack((np.array(pu_m), np.array(pv_m), np.array(d_x2), np.array(d_y2), np.array(peakCorr), np.array(meanAbsCorr), np.array(SNratio)), axis=1)

                # Headers
                line1 = 'This table has been created from the software IDMatch and contains results from image matching'
                line2 = ''.join(['The results below were generated with the matching function M1 and window size ', str(template_size)])
                line3 = ''
                line4 = 'Variables:'
                line5 = 'x1, y1 = x- and y- the coordinates of a point matched (p) in image 1 (in pixel)'
                line6 = 'dx2, dy2 = the displacement in x- and y-direction from a point in image 1 to that point in image 2 (in pixel)'
                line7 = 'peakCorr = The peak correlation found for each p in the search window in image 2'
                line8 = 'meanAbsCorr = The mean absolute correlation calculated over the whole search window in image 2'
                line9 = 'SNratio = Signal to noise ratio between the peakCorr and meanAbsCorr'
                line10 = ''
                line11 = 'x1 y1 dx2 dy2 peakCorr meanAbsCorr SNratio'
                header_matchtable = '\n'.join([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11])
                np.savetxt(table_base_name, table_array, fmt='%i\t%i\t%1.4f\t%1.4f\t%1.4f\t%1.4f\t%1.4f', newline='\n', header=header_matchtable, delimiter='\t', comments='# ', encoding=None)


                #--- Calculate time
                end_time_win = time.time()  # time after one window
                if count_pairs == 1 and count_win == 1:
                    est_time = ((end_time_win-start_time_win) * len(match_win_size) * len(files1) * len(method_list)) - (end_time_win-start_time_win)  # = time_1win * nbr_win * nbr_pair * nbr_methods - (1* time_1win)
                    print("Estimated remaining time for matching (h:mm:ss): ", str(datetime.timedelta(seconds=est_time)))
                # ---

                count_win = count_win + 1  # counter for nbr of windows


            # --- Calculate time
            if count_pairs == 1 and count_win == len(match_win_size):
                end_time_pair = time.time()  # time after 1 pair
                est_time = ((end_time_pair - start_time_pair) * len(files1) * len(method_list)) - (end_time_pair - start_time_pair)  # = time_1pair * nbr_pair * nbr_methods - (1* time_1win)
                print("Estimated remaining time for matching (h:mm:ss): ", str(datetime.timedelta(seconds=est_time)))
            # ---

            count_pairs = count_pairs + 1  # counter for image pairs

            # end loop over all dataset pairs

        # --- Calculate time
        end_time_M1 = time.time()  # time after one matching method
        if len(method_list) > 1:
            est_time = (end_time_M1 - start_time_M1) * (len(method_list)-1)
            print("Estimated remaining time for matching (h:mm:ss): ", str(datetime.timedelta(seconds=est_time)))
        # ---
    # end method 1


    # Matching method 2 (Orientation Correlation (OC))
    if 'M2' in method_list:
        print("Matching method 2")
        start_time_M2 = time.time()

        count_pairs = 1
        start_time_pair = time.time()
        for f1, f2 in zip(files1, files2):
            file_names = os.path.splitext(os.path.basename(f1))[0]
            file_names = file_names[:1] + file_names[2:]

            # Open image pair
            img1 = gdal.Open(f1, gdalconst.GA_ReadOnly)
            img2 = gdal.Open(f2, gdalconst.GA_ReadOnly)
            img1_array = img1.ReadAsArray().astype(np.float)
            img2_array = img2.ReadAsArray().astype(np.float)
            img1_array = img1_array / 255
            img2_array = img2_array / 255

            # Filters
            f = np.array([1, 0, -1]).reshape((1,3))
            ff = np.vstack([-1, 0, 1])

            # calculate x and y orientation gradients
            img1_gradx = scipy.ndimage.correlate(img1_array, ff, mode='nearest')
            img1_grady = scipy.ndimage.correlate(img1_array, f, mode='nearest')
            img2_gradx = scipy.ndimage.correlate(img2_array, ff, mode='nearest')
            img2_grady = scipy.ndimage.correlate(img2_array, f, mode='nearest')

            img1_orient = np.exp(1j * np.arctan2(img1_grady, img1_gradx))
            img2_orient = np.exp(1j * np.arctan2(img2_grady, img2_gradx))
            img2_orient = np.conj(img2_orient)

            start_time_win = time.time()

            count_win = 1  # counter for nbr of windows
            for template_size in match_win_size:  # Loop over each template size  template_size = match_win_size[0]
                print("Data pair: ", count_pairs, "/", np.size(files1), "Window: ", count_win, "/", np.size(match_win_size))
                search_window = template_size + (pix_dev*2)  # *2 because the displacement can take place in all directions

                htemp_size = int(template_size / 2)
                hsearch_win = int(search_window / 2)
                proc_size = template_size + search_window - 1

                # padding arrays for moving window
                img1_pad = np.pad(img1_orient, [(hsearch_win,), (hsearch_win,)], 'constant', constant_values=0.0)
                img2_pad = np.pad(img2_orient, [(hsearch_win,), (hsearch_win,)], 'constant', constant_values=0.0)

                # correct pv_m and pu_m values based on padding
                pu_m = [x + hsearch_win for x in pu_m]
                pv_m = [x + hsearch_win for x in pv_m]

                peakCorr = np.empty(len(pv_m)) * np.nan
                meanAbsCorr = np.empty(len(pv_m)) * np.nan
                d_x2 = np.empty(len(pv_m)) * np.nan
                d_y2 = np.empty(len(pv_m)) * np.nan
                for i in range(len(pv_m)):  # Make moving template on selected points 'p', which is the center point of the template and search_win
                    template = img1_pad[pv_m[i]-htemp_size: pv_m[i]+(htemp_size+1), pu_m[i]-htemp_size: pu_m[i]+(htemp_size+1)]  # +1 to put the p in the center
                    search_win = img2_pad[pv_m[i]-hsearch_win : pv_m[i]+(hsearch_win+1), pu_m[i]-hsearch_win: pu_m[i]+(hsearch_win+1)]

                    pad = int((search_win.shape[0] - template.shape[0]) /2)

                    fft_template = np.fft.fft2(np.rot90(template, 2), s=(proc_size, proc_size))
                    fft_search = np.fft.fft2(search_win, s=(proc_size, proc_size))
                    result = np.real(np.fft.ifft2(fft_template * fft_search))

                    # Cropping results not affected by edges
                    result_nopad = result[np.int((proc_size / 2) - pad): np.int((proc_size / 2) + (pad + 1)), np.int((proc_size / 2) - pad): np.int((proc_size / 2) + (pad + 1))]
                    x_res = np.array(range(-int(len(result_nopad) / 2), int(len(result_nopad) / 2) + 1))
                    y_res = range(-int(len(result_nopad) / 2), int(len(result_nopad) / 2) + 1)

                    # Get coordinates of max value and max value
                    y_m, x_m = np.unravel_index(np.argmax(result_nopad), result_nopad.shape)  # row, col as int.
                    max_val = np.max(result_nopad)

                    ## Sub-pixel accuracy
                    # Check whether the peak is on an edge
                    edgedist = np.min(np.abs([0 - y_m, 0 - x_m, y_m - np.shape(result_nopad)[0], x_m - np.shape(result_nopad)[0]]))

                    if edgedist == 0:  # don't take this point 'p' into account
                        subpix_deviation_y = np.nan
                        subpix_deviation_x = np.nan

                    elif edgedist == 1:
                        win_subpix = 3
                        hwin_subpix = int(win_subpix / 2)  # 3x3 window

                        # Take part of the correlation around the max peak
                        result_win = result_nopad[y_m - hwin_subpix: y_m + (hwin_subpix + 1), x_m - hwin_subpix: x_m + (hwin_subpix + 1)]
                        y_sub, x_sub = np.meshgrid(y_res[y_m - hwin_subpix: y_m + (hwin_subpix + 1)], x_res[x_m - hwin_subpix: x_m + (hwin_subpix + 1)], indexing='ij')

                        result_win_flat = np.ndarray.flatten(result_win)
                        middle = int((len(result_win_flat) - 1) / 2)
                        np.delete(result_win_flat, middle)
                        result_win_mean = np.nanmean(result_win_flat)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            result_win = result_win - result_win_mean
                            result_win[result_win < 0] = 0
                            result_win = result_win / np.sum(result_win)

                        subpix_deviation_y = np.sum(y_sub * result_win)
                        subpix_deviation_x = np.sum(x_sub * result_win)

                    else:
                        win_subpix = 5; hwin_subpix = int(win_subpix / 2)  # !!!advanced_param: Here the size of the window from which subpixel displacement is derived can be changed. default = 5x5 window

                        # Take part of the correlation around the max peak
                        result_win = result_nopad[y_m - hwin_subpix: y_m + (hwin_subpix + 1), x_m - hwin_subpix: x_m + (hwin_subpix + 1)]
                        y_sub, x_sub = np.meshgrid(y_res[y_m - hwin_subpix: y_m + (hwin_subpix + 1)], x_res[x_m - hwin_subpix: x_m + (hwin_subpix + 1)], indexing='ij')

                        result_win_flat = np.ndarray.flatten(result_win)
                        middle = int((len(result_win_flat) - 1) / 2)
                        np.delete(result_win_flat, middle)
                        result_win_mean = np.nanmean(result_win_flat)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            result_win = result_win - result_win_mean
                            result_win[result_win < 0] = 0
                            result_win = result_win / np.sum(result_win)

                        subpix_deviation_y = np.sum(y_sub * result_win)
                        subpix_deviation_x = np.sum(x_sub * result_win)

                    # Save values for point 'p'
                    if subpix_deviation_y == np.nan:
                        peakCorr[i] = np.nan
                        meanAbsCorr[i] = np.nan
                        d_y2[i] = subpix_deviation_y  # rows
                        d_x2[i] = subpix_deviation_x  # cols
                    else:
                        peakCorr[i] = max_val
                        meanAbsCorr[i] = np.nanmean(np.abs(result_nopad))
                        d_y2[i] = subpix_deviation_y
                        d_x2[i] = subpix_deviation_x

                # end for loop going over all points 'p'

                try:
                    SNratio = peakCorr / meanAbsCorr
                except RuntimeWarning:
                    SNratio = np.nan

                # Reset correct pv_m and pu_m values without padding
                pu_m = [x - hsearch_win for x in pu_m]
                pv_m = [x - hsearch_win for x in pv_m]

                ##--- SAVE RESULTS FROM MATCHING ---
                table_base_name = res_matching + '\\' + file_names + 'M2W' + str(template_size).zfill(4) + '.txt'
                table_array = np.stack((np.array(pu_m), np.array(pv_m), np.array(d_x2), np.array(d_y2), np.array(peakCorr), np.array(meanAbsCorr), np.array(SNratio)), axis=1)

                # Headers
                line1 = 'This table has been created from the software IDMatch and contains results from image matching'
                line2 = ''.join(['The results below were generated with the matching function M2 and window size ', str(template_size)])
                line3 = ''
                line4 = 'Variables:'
                line5 = 'x1, y1 = x- and y- the coordinates of a point matched (p) in image 1 (in pixel)'
                line6 = 'dx2, dy2 = the displacement in x- and y-direction from a point in image 1 to that point in image 2 (in pixel)'
                line7 = 'peakCorr = The peak correlation found for each p in the search window in image 2'
                line8 = 'meanAbsCorr = The mean absolute correlation calculated over the whole search window in image 2'
                line9 = 'SNratio = Signal to noise ratio between the peakCorr and meanAbsCorr'
                line10 = ''
                line11 = 'x1 y1 dx2 dy2 peakCorr meanAbsCorr SNratio'
                header_matchtable = '\n'.join([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11])
                np.savetxt(table_base_name, table_array, fmt='%i\t%i\t%1.4f\t%1.4f\t%1.4f\t%1.4f\t%1.4f', newline='\n', header=header_matchtable, delimiter='\t', comments='# ', encoding=None)


                #--- timing
                end_time_win = time.time()  # time after one window
                if count_pairs == 1 and count_win == 1:  # print remaining time only afer the first window loop
                    est_time = ((end_time_win-start_time_win) * len(match_win_size) * len(files1) * len(method_list)) - (end_time_win-start_time_win)  # = time_1win * nbr_win * nbr_pair * nbr_methods - (1* time_1win)
                    print("Estimated remaining time for matching (h:mm:ss): ", str(datetime.timedelta(seconds=est_time)))
                # ---

                count_win = count_win + 1  # counter for nbr of windows

            # end of loop over al windows

            # --- timing
            if count_pairs == 1 and count_win == len(match_win_size):
                end_time_pair = time.time()  # time after 1 pairs
                est_time = ((end_time_pair - start_time_pair) * len(files1) * len(method_list)) - (end_time_pair - start_time_pair)  # = time_1pair * nbr_pair * nbr_methods - (1* time_1win)
                print("Estimated remaining time for matching (h:mm:ss): ", str(datetime.timedelta(seconds=est_time)))
            # ---

            count_pairs = count_pairs + 1  # counter for image pairs

            # end of loop over all dataset pairs


        # --- timing
        end_time_M2 = time.time()  # time after one matching method
        if len(method_list) > 1:
            est_time = (end_time_M2 - start_time_M2) * (len(method_list)-1)
            print("Estimated remaining time for matching (h:mm:ss): ", str(datetime.timedelta(seconds=est_time)))
        # ---

    # end method 2

    # Matching method 3 (SURF+ BF)
    if 'M3' in method_list:
        print("Matching method 3")
        start_time_M3 = time.time()

        files1 = [i for i in files1 if 'H1' in i or 'I1' in i]
        files2 = [i for i in files2 if 'H2' in i or 'I2' in i]

        count_pairs = 1
        start_time_pair = time.time()
        for f1, f2 in zip(files1, files2):
            file_names = os.path.splitext(os.path.basename(f1))[0]
            file_names = file_names[:1] + file_names[2:]

            # Open images as grayscale
            im1 = cv2.imread(f1, 0)
            im2 = cv2.imread(f2, 0)

            start_time_win = time.time()
            count_win = 1  # counter for nbr of windows
            for template_size in match_win_size:  # Loop over each template size  template_size = match_win_size[0]
                print("Data pair: ", count_pairs, "/", np.size(files1), "Window: ", count_win, "/", np.size(match_win_size))
                search_window = template_size + (pix_dev*2)  # *2 because the displacement can take place in all directions
                htemp_size = int(template_size / 2)
                hsearch_win = int(search_window / 2)

                # Padding arrays for moving window
                img1_pad = np.pad(im1, [(hsearch_win,), (hsearch_win,)], 'constant', constant_values=255)
                img2_pad = np.pad(im2, [(hsearch_win,), (hsearch_win,)], 'constant', constant_values=255)

                # Correct pv_m and pu_m values based on padding
                pu_m = [x + hsearch_win for x in pu_m]
                pv_m = [x + hsearch_win for x in pv_m]

                d_x2 = np.empty(len(pv_m)) * np.nan
                d_y2 = np.empty(len(pv_m)) * np.nan
                for i in range(len(pv_m)):  # Make moving template on selected points 'p', which is the center point of the template and search_win
                    template = img1_pad[pv_m[i] - htemp_size: pv_m[i] + (htemp_size + 1), pu_m[i] - htemp_size: pu_m[i] + (htemp_size + 1)]  # +1 to put the p in the center
                    search_win = img2_pad[pv_m[i] - hsearch_win: pv_m[i] + (hsearch_win + 1), pu_m[i] - hsearch_win: pu_m[i] + (hsearch_win + 1)]

                    # Pad template array
                    template = np.pad(template, [(pix_dev,), (pix_dev,)], 'constant', constant_values=np.nan)

                    # Create SURF object and find keypoints and descriptors
                    hess_thres = 300  # Hessian Threshold (best between 300 and 500). The largest the values, the fewer the kp/des
                    surf1 = cv2.xfeatures2d.SURF_create(hess_thres, upright=0, extended=1)
                    kp1, des1 = surf1.detectAndCompute(template, None)
                    surf2 = cv2.xfeatures2d.SURF_create(hess_thres, upright=0, extended=1)
                    kp2, des2 = surf2.detectAndCompute(search_win, None)

                    # Match descriptor vectors using Brute Force
                    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

                    if len(kp1) != 0 and len(kp2) != 0:
                        good_matches = bf.match(des1, des2)

                        # Alternative with flann:
                        # matches = flann.knnMatch(des1, des2, k=2)
                        # good_matches = [m for (m, n) in matches if m.distance < 0.7 * n.distance]  # ratio test as per Lowe's paper

                        # Calculate pixel deviation between the features from each epoch
                        if good_matches != []:
                            matches_s = sorted(good_matches, key=lambda x: x.distance)
                            (x1_keep, y1_keep) = kp1[matches_s[0].queryIdx].pt  # x=cols, y=rows
                            (x2_keep, y2_keep) = kp2[matches_s[0].trainIdx].pt

                            d_x2[i] = x2_keep - x1_keep
                            d_y2[i] = y2_keep - y1_keep
                        else:
                            d_x2[i] = np.nan
                            d_y2[i] = np.nan
                    else:
                        # If the number of keypoints is 0 in reference or search template, return np.nan. These points will later be discarded.
                        d_x2[i] = np.nan
                        d_y2[i] = np.nan

                # end loop around each 'p'

                # Reset correct pv_m and pu_m values without padding
                pu_m = [x - hsearch_win for x in pu_m]
                pv_m = [x - hsearch_win for x in pv_m]

                # No Signal-to-Noise for this method
                peakCorr = np.empty(len(pv_m)) * np.nan
                meanAbsCorr = np.empty(len(pv_m)) * np.nan
                SNratio = np.empty(len(pv_m)) * np.nan

                ##--- SAVE RESULTS FROM MATCHING ---
                table_base_name = res_matching + '\\' + file_names + 'M3W' + str(template_size).zfill(4) + '.txt'
                table_array = np.stack((np.array(pu_m), np.array(pv_m), np.array(d_x2), np.array(d_y2), np.array(peakCorr), np.array(meanAbsCorr), np.array(SNratio)), axis=1)

                # Headers
                line1 = 'This table has been created from the software IDMatch and contains results from image matching'
                line2 = ''.join(['The results below were generated with the matching function M3 and window size ',str(template_size)])
                line3 = ''
                line4 = 'Variables:'
                line5 = 'x1, y1 = x- and y- the coordinates of a point matched (p) in image 1 (in pixel)'
                line6 = 'dx2, dy2 = the displacement in x- and y-direction from a point in image 1 to that point in image 2 (in pixel)'
                line7 = 'peakCorr = The peak correlation found for each p in the search window in image 2'
                line8 = 'meanAbsCorr = The mean absolute correlation calculated over the whole search window in image 2'
                line9 = 'SNratio = Signal to noise ratio between the peakCorr and meanAbsCorr'
                line10 = ''
                line11 = 'x1 y1 dx2 dy2 peakCorr meanAbsCorr SNratio'
                header_matchtable = '\n'.join([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11])
                np.savetxt(table_base_name, table_array, fmt='%i\t%i\t%1.4f\t%1.4f\t%1.4f\t%1.4f\t%1.4f', newline='\n', header=header_matchtable, delimiter='\t', comments='# ', encoding=None)

                # --- timing
                end_time_win = time.time()  # time after one window
                if count_pairs == 1 and count_win == 1:
                    est_time = ((end_time_win - start_time_win) * len(match_win_size) * len(files1) * len(method_list)) - (end_time_win - start_time_win)  # = time_1win * nbr_win * nbr_pair * nbr_methods - (1* time_1win)
                    print("Estimated remaining time for matching (h:mm:ss): ", str(datetime.timedelta(seconds=est_time)))
                # ---

                count_win = count_win + 1  # counter for nbr of windows

            # --- timing
            if count_pairs == 1 and count_win == len(match_win_size):
                end_time_pair = time.time()  # time after 1 pairs
                est_time = ((end_time_pair - start_time_pair) * len(files1) * len(method_list)) - (end_time_pair - start_time_pair)  # = time_1pair * nbr_pair * nbr_methods - (1* time_1win)
                print("Estimated remaining time for matching (h:mm:ss): ", str(datetime.timedelta(seconds=est_time)))
            # ---

            count_pairs = count_pairs + 1  # counter for image pairs
        # end of loop over all dataset pairs

        # --- timing
        end_time_M3 = time.time()  # time after one matching method
        if len(method_list) > 1:
            est_time = (end_time_M3 - start_time_M3) * (len(method_list)-1)
            print("Estimated remaining time for matching (h:mm:ss): ", str(datetime.timedelta(seconds=est_time)))
        # ---

    # end method 3

    return [pv_m, pu_m]


def postfilter(val_pts_option, res_path, res_matching, res_postfilt, step_grid, min_win_members, times_stepgrid, data_info, tot_match_pts):
    """
    This function filters the vectors obtained from the matching function.

       Parameters
       ----------

       val_pts_option : str
           Path to .txt file that contains validation points if they are available
       res_path : str
           Path to folder that holds all results from IDMatch
       res_matching : str
            Path to the folder where the results generated from the matching function are stored (temporary folder)
       res_postfilt : str
            Path to the folder where the results generated from the post-filtering procedure will be stored
       step_grid : int
            Spacing chosen by the user in the params.py file that defines when a matching measurement took place. The number is in pixel
       min_win_members : int
            Minimum number of points inside the post-filtering window. These points are used to calculate the mean displacement and the std
       times_stepgrid : int
            Defines the size of the post-filtering window. times_stepgrid * step_grid
       data_info : list
            List containing the coordinates of dataset origin in the x- and y-dimension, the pixel width and height, number of rows and columns of the image as well as the projection. In order: xOrigin, pixelWidth, yOrigin, pixelHeight, rows, cols, proj
       tot_match_pts : list
            List of pixel coordinates (rows, columns) of all points 'p' that were matched


       Returns
       -------



       """

    ## Message:
    print("## Post-filtering")

    ## List all matching tables
    tables_list = os.listdir(res_matching)
    table_list = []; file_names = []
    for file in tables_list:
        if file.endswith("txt"):
            table_list.append(os.path.join(res_matching, file))
            file_names.append(os.path.splitext(file)[0])
        else:
            raise Exception('The imported filed need to be .txt')
    if len(tables_list) == 0:
        raise Exception('There is no matching tables in path_tables_match')
    table_list.sort(); file_names.sort()

    ## Initialization

    # Get information on dataset size
    xOrigin, pixelWidth, yOrigin, pixelHeight, rows, cols, proj = data_info  # x ans y Origin is in the top left

    # Create result table
    result_postfilt_table = np.empty((10, 0))
    stack_magn = np.empty((len(tot_match_pts[0]), len(tables_list))) * np.nan  # makes a 3D array
    stack_angle = np.empty((len(tot_match_pts[0]), len(tables_list))) * np.nan  # makes a 3D array

    if val_pts_option != "":
        val_x1, val_y1, val_x2, val_y2 = np.loadtxt(val_pts_option, skiprows=1, unpack=True, delimiter="\t")  # need to be x1,y1, x2,y2
        result_validation_table = np.empty((len(val_x1), 0))
        result_validation_table2 = np.empty((0, 4))  # 4cols for RMSE and R2 (magn and angle)

    ##--- LOOP OVER ALL MATCHING TABLES ---
    count_table = 1
    start_time = time.time()
    for table in table_list:
        print("table: ", count_table, "/", len(table_list))

        # --- Import matching table
        x1, y1, d_x2, d_y2, peakCorr, meanAbsCorr, SNratio = np.loadtxt(table, skiprows=11, unpack=True, delimiter='\t')  # x1, y1 = pixel coord of point 1; d_x2, d_y2 the deviation in x and y dimension between point 1 and point 2
        x1 = x1.astype(int)
        y1 = y1.astype(int)

        # Calculate the magnitude and angle
        magn = np.sqrt((d_x2 ** 2) + (d_y2 ** 2))  # in pixels
        angle = np.arctan2(d_y2, d_x2)  # angle in radians
        with warnings.catch_warnings():  # several points 'p' in the matching table can be np.nan, raising error.
            warnings.simplefilter("ignore")
            angle = np.degrees(angle) % 360  # angle in degrees (0-360°)

        # Re-create vx, vy, length and angle 2D arrays from table
        vx = np.full((rows, cols), np.nan)
        vy = np.full((rows, cols), np.nan)
        magnitude = np.full((rows, cols), np.nan)
        alpha = np.full((rows, cols), np.nan)
        for i in range(len(x1)):
            vx[y1[i], x1[i]] = d_x2[i]  # distance of vector between p1 and p2 in the x direction (cols)
            vy[y1[i], x1[i]] = d_y2[i]  # distance of vector between p1 and p2 in the y direction (rows)
            magnitude[y1[i], x1[i]] = magn[i]  # distance of vector between p1 and p2
            alpha[y1[i], x1[i]] = angle[i]  # angle in degree between p1 and p2

        ##--- MOVING WINDOWS ---

        # --- Initialization

        # Set moving window size
        wind_size = step_grid * times_stepgrid  # define post-filtering window size
        if wind_size % 2 == 0:
            wind_size = wind_size + 1  # needs to be odd to have a center
        hwind_size = int(wind_size/2)

        # Set counting lists and arrays
        pts_notmatch = 0; filt_magn = 0; filt_dir = 0; filt_snr = 0; filt_memb = 0  # counters for the number of mismatches
        point_nokeep = np.full((rows, cols), np.nan)  # array that keeps track of the points we keep (with zeros) and the ones we don't (with values)
        count_win_el = np.full(len(x1), np.nan)  # array that stores the nbr of elements in each window

        # Padding arrays with np.nan
        vx_pad = np.pad(vx, [(hwind_size, ), (hwind_size, )], 'constant', constant_values=np.nan)
        vy_pad = np.pad(vy, [(hwind_size, ), (hwind_size, )], 'constant', constant_values=np.nan)
        alpha_pad = np.pad(alpha, [(hwind_size, ), (hwind_size, )], 'constant', constant_values=np.nan)
        magnitude_pad = np.pad(magnitude, [(hwind_size, ), (hwind_size, )], 'constant', constant_values=np.nan)

        id_notnan = list(np.where(~np.isnan(d_x2)))  # there a points 'p' that were not matched (np.nan) and that will therefore not be post-filtered
        # --- Start moving windows
        for i in range(len(x1)):  # make window on each matched point 'p'
            if i in id_notnan[0]:  # check if point 'p' was matched

                # Correct y1 and x1 pixel coordinates (point 'p') based on padding
                py, px = y1[i] + hwind_size, x1[i] + hwind_size
                point_nokeep[y1[i], x1[i]] = np.nansum([point_nokeep[y1[i], x1[i]], 0])

                win_vx = vx_pad[py-hwind_size:py+(hwind_size+1), px-hwind_size:px+(hwind_size+1)]  # +1 to put the point 'p' in the center
                win_vy = vy_pad[py-hwind_size:py+(hwind_size+1), px-hwind_size:px+(hwind_size+1)]
                win_alpha = alpha_pad[py-hwind_size:py+(hwind_size+1), px-hwind_size:px+(hwind_size+1)]
                win_magn = magnitude_pad[py-hwind_size:py+(hwind_size+1), px-hwind_size:px+(hwind_size+1)]

                nbr_win_members = np.sum(~np.isnan(win_vx))
                count_win_el[i] = nbr_win_members
                if nbr_win_members >= min_win_members:

                    # Initializing array recording where filters are taking place
                    point_nokeep[y1[i], x1[i]] = np.nansum([point_nokeep[y1[i], x1[i]], 0])

                    win_magn = win_magn[np.where(~np.isnan(win_magn))]
                    magn_neighbours = win_magn.flatten()

                    win_alpha2 = win_alpha[np.where(~np.isnan(win_alpha))]
                    alpha_neighbours = win_alpha2.flatten()

                    ## Magnitude filter

                    # Calculate the magnitude difference between point 'p' and its neighbours
                    magn_max = 2 / pixelWidth  # !!!advanced_param: maximum percentage that the matched points can have with their neighbouring points
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        magn_diff = np.abs(magn_neighbours - magnitude_pad[py, px])
                        magn_idx = np.where(magn_diff <= magn_max)
                        nbr_similar_magn_neighbours = magn_neighbours[magn_idx]

                    # Calculate percentage of similar neighbours (neighbours that have a difference in magnitude smaller than magn_max)
                    try:
                        percent_sim_neighbours = len(nbr_similar_magn_neighbours) / len(magn_neighbours)
                    except ZeroDivisionError:
                        percent_sim_neighbours = 0


                    std_nbr_magn = 3  # !!!advanced_param: number of standard deviation used for setting the filtering threshold for the directional filter
                    percent_neighbours = 0.5  # !!!advanced_param: if a point 'p' has less than 50% (0.5 = default) neighbours that have an angle differing of more than 'angle_max'.
                    if np.abs(vx_pad[py, px] - np.nanmedian(win_vx)) > (std_nbr_magn*np.nanstd(win_vx)) or np.abs(vy_pad[py, px] - np.nanmedian(win_vy)) > (std_nbr_magn*np.nanstd(win_vy)) or percent_sim_neighbours <= percent_neighbours:
                        point_nokeep[y1[i], x1[i]] = np.nansum([point_nokeep[y1[i], x1[i]], 1])  # that a non-padded array size so the original pixel coordinates are used
                        filt_magn = filt_magn + 1
                    else:
                        point_nokeep[y1[i], x1[i]] = np.nansum([point_nokeep[y1[i], x1[i]], 0])

                    ## Directional filter

                    # Select the non-nan values to calculate sd
                    idy, idx = np.where(~np.isnan(win_alpha))
                    values_angle_wind_all = win_alpha[idy, idx]
                    std_nbr_dir = 3  # !!!advanced_param: number of standard deviation used for setting the filtering threshold for the directional filter

                    # Calculate mean angle of the window ('p' included)
                    win_alpha_sin_mean = np.nanmean(np.sin(np.radians(win_alpha)))
                    win_alpha_cos_mean = np.nanmean(np.cos(np.radians(win_alpha)))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        win_tan_mean = np.arctan2(win_alpha_sin_mean, win_alpha_cos_mean)  # in radians
                        win_angle_mean = np.degrees(win_tan_mean) % 360  # in angles (0-360°)

                        # Calculate std of the window ('p' included)
                        abs_diff_2 = np.abs(((values_angle_wind_all - win_angle_mean) + 180) % 360 - 180)**2
                        win_angle_std = np.sqrt(np.nansum(abs_diff_2)/np.size(values_angle_wind_all))


                    # Calculate the angle difference between point 'p' and its neighbours
                    angle_max = 20  # !!!advanced_param: maximum angle that the matched points can have with their neighbouring points
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        anglediff = np.abs(((alpha_neighbours - alpha_pad[py, px]) + 180) % 360 - 180)
                        angle_idx = np.where(anglediff <= angle_max)
                        nbr_similar_alpha_neighbours = alpha_neighbours[angle_idx]

                    # Calculate percentage of similar neighbours (neighbours that have a difference in angle smaller than angle_max)
                    try:
                        percent_sim_neighbours = len(nbr_similar_alpha_neighbours) / len(alpha_neighbours)
                    except ZeroDivisionError:
                        percent_sim_neighbours = 0

                    # Calculate the angle difference between point 'p' and the mean of the window
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        diff_angle_p_win = np.abs(((alpha_pad[py, px] - win_angle_mean) + 180) % 360 - 180)

                    percent_neighbours = 0.5  # !!!advanced_param: if a point 'p' has less than 50% (0.5 = default) neighbours that have an angle differing of more than 'angle_max'.
                    if diff_angle_p_win > (std_nbr_dir*win_angle_std) or percent_sim_neighbours <= percent_neighbours:
                        point_nokeep[y1[i], x1[i]] = np.nansum([point_nokeep[y1[i], x1[i]], 2])
                        filt_dir = filt_dir + 1
                    else:
                        point_nokeep[y1[i], x1[i]] = np.nansum([point_nokeep[y1[i], x1[i]], 0])


                    ## Signal-to-Noise ratio filter
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        SNratio_normed = (SNratio - np.min(SNratio)) / (np.max(SNratio) - np.min(SNratio))  # normalized between 0 (bad signal) and 1 (good signal)

                    snr_threshold = 0.2  # !!!advanced_param: threshold for the Signal-to-Noise Ratio (normalized).

                    if SNratio_normed[i] < snr_threshold:
                        point_nokeep[y1[i], x1[i]] = np.nansum([point_nokeep[y1[i], x1[i]], 4])
                        filt_snr = filt_snr + 1
                    else:
                        point_nokeep[y1[i], x1[i]] = np.nansum([point_nokeep[y1[i], x1[i]], 0])

                    # end of magnitude, directional and SNR filters

                else:  # if the nbr_win_members < min_win_members:
                    point_nokeep[y1[i], x1[i]] = np.nansum([point_nokeep[y1[i], x1[i]], 8])  # the points that do not have enough members are filtered out
                    filt_memb = filt_memb + 1

                # end loop of the window_members

            else:  # if the point 'p' has not been matched
                point_nokeep[y1[i], x1[i]] = np.nansum([point_nokeep[y1[i], x1[i]], 16])
                pts_notmatch = pts_notmatch + 1


        # end running windows
        # --------------------


        ## After the filters for each point 'p' (above), several additional filters are applied on the filtered velocity field

        magnitude_filt = np.empty(np.shape(magn)) * np.nan
        angle_filt = np.empty(np.shape(angle)) * np.nan
        id_good2D = np.where(point_nokeep == 0)  # index of the points that are kept, in (row, col)

        # Check if all points are filtered with the filters above
        if np.nansum(id_good2D) == 0:
            magnitude_filt = magnitude_filt
            angle_filt = angle_filt
            pts_filtered = point_nokeep[y1, x1].flatten()
            id_good = []
            # magn_mean, magn_std, win_angle_mean, win_angle_std = np.nan, np.nan, np.nan, np.nan
        else:

            ## Filter point threshold
            tot_pts_percent = 0.15  # !!!advanced_param: if the percentage of left points after all above filtering process is below tot_pts_percent, it means that the datasets are probably not good and the remaining points neither.
            filt_patch = 0
            if len(id_good2D[0]) < tot_pts_percent * len(tot_match_pts[0]):
                point_nokeep[id_good2D] = np.nansum([point_nokeep[id_good2D], 32])
                magnitude_filt = magnitude_filt
                angle_filt = angle_filt
                pts_filtered = point_nokeep[y1, x1].flatten()
                id_good = []
                filt_patch = len(id_good2D[0])
                # magn_mean, magn_std, win_angle_mean, win_angle_std = np.nan, np.nan, np.nan, np.nan

            else:  # if the number of points left are enough, they are going through a second round of filters
                filt_patch = 0
                magnitude2 = np.empty(np.shape(magnitude)) * np.nan
                magnitude2[id_good2D] = magnitude[id_good2D]  # magnitude array (from matching) filtered

                # Single direction filter
                angle2 = np.empty(np.shape(alpha)) * np.nan
                angle2[id_good2D] = alpha[id_good2D]  # angle array (from matching) filtered

                win_alpha_sin_mean = np.nanmean(np.sin(np.radians(angle2)))
                win_alpha_cos_mean = np.nanmean(np.cos(np.radians(angle2)))
                win_tan_mean = np.arctan2(win_alpha_sin_mean, win_alpha_cos_mean)
                win_angle_mean = np.degrees(win_tan_mean) % 360

                single_dir = 70  # !!!advanced_param:
                for iid in range(0, len(id_good2D[0])):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        anglediff = np.abs(((win_angle_mean - angle2[id_good2D[0][iid], id_good2D[1][iid]]) + 180) % 360 - 180)
                        if anglediff > single_dir:
                            point_nokeep[id_good2D[0][iid], id_good2D[1][iid]] = np.nansum([point_nokeep[id_good2D[0][iid], id_good2D[1][iid]], 2])
                            filt_dir = filt_dir + 1

        # end of secondary filters

        # Find final indexes of good points 'p', and transform the point_nokeep in 1D array
        pts_filtered = point_nokeep[y1, x1].flatten()
        id_good = np.where(pts_filtered == 0)  # indices in 1D
        id_good2D = np.where(point_nokeep == 0)  # indices in 2D

        magnitude_filt[id_good] = magn[id_good]  # magnitude array (from matching) filtered
        angle_filt[id_good] = angle[id_good]  # angle array (from matching) filtered

        # # Calculate mean and std of magnitude and angles for each combination
        # if np.nansum(id_good) != 0:
        #     magn_mean = np.nanmean(magnitude_filt)
        #     magn_std = np.nanstd(magnitude_filt)
        #     win_alpha_sin_mean = np.nanmean(np.sin(np.radians(angle_filt)))
        #     win_alpha_cos_mean = np.nanmean(np.cos(np.radians(angle_filt)))
        #     win_tan_mean = np.arctan2(win_alpha_sin_mean, win_alpha_cos_mean)
        #     win_angle_mean = np.degrees(win_tan_mean) % 360
        #     abs_diff_2 = np.abs(((angle_filt[id_good] - win_angle_mean) + 180) % 360 - 180) ** 2
        #     win_angle_std = np.sqrt((np.nansum(abs_diff_2)) / (len(id_good[0])))
        # else:
        #     magn_mean, magn_std, win_angle_mean, win_angle_std = np.nan, np.nan, np.nan, np.nan

        # Save points left after filtering. Stack all tables in these arrays
        stack_magn[:, (count_table - 1)] = magnitude_filt
        stack_angle[:, (count_table - 1)] = angle_filt


        ##--- SAVE NEW INFORMATION TO THE MATCHING TABLE ---
        results_matching_filtered = np.column_stack((x1, y1, d_x2, d_y2, peakCorr, meanAbsCorr, SNratio, magn, angle, pts_filtered))

        table_base_name = os.path.splitext(os.path.basename(table))[0]
        table_path_filt = res_postfilt + '\\' + table_base_name + '.txt'

        idxW = table_base_name.find('W')
        template_size = table_base_name[(idxW + 1):len(table_base_name)]

        # Headers
        line1 = 'This table has been created from the software IDMatch and contains results from image matching'
        line2 = ''.join(['The results below were generated with the matching function Mx and window size ', str(template_size)])
        line3 = ''
        line4 = 'Variables:'
        line5 = 'x1, y1 = x- and y- the coordinates of a point matched (p) in image 1 (in pixel)'
        line6 = 'dx2, dy2 = the displacement in x- and y-direction from a point in image 1 to that point in image 2 (in pixel)'
        line7 = 'peakCorr = The peak correlation found for each p in the search window in image 2'
        line8 = 'meanAbsCorr = The mean absolute correlation calculated over the whole search window in image 2'
        line9 = 'SNratio = The signal-to-noise ratio, which is the ration between the peakCorr and the meanAbsCorr.'
        line10 = 'magn = The magnitude or the displacement between p in image 1 and p in image 2 (in pixel)'
        line11 = 'angle = The angle of the vector relying p in image 1 and p in image 2 (in degree)'
        line12 = 'pts_filtered = The point that are kept after filtering (0) and that are discarded (with positive number)'
        line13 = ''
        line14 = 'x1 y1 dx2 dy2 peakCorr meanAbsCorr SNratio magn angle pts_filtered'
        header_matchtable = '\n'.join([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13, line14])
        if os.path.exists(table_path_filt): os.remove(table_path_filt)
        np.savetxt(table_path_filt, results_matching_filtered, fmt='%i\t%i\t%1.4f\t%1.4f\t%1.4f\t%1.4f\t%1.4f\t%1.4f\t%1.4f\t%i', newline='\n',header=header_matchtable, delimiter='\t', comments='#', encoding=None)

        # Fill post-filter result table
        tot_matched = (len(tot_match_pts[0])-pts_notmatch) / len(tot_match_pts[0])  # total points that were matched
        tot_m = len(tot_match_pts[0])-pts_notmatch

        if id_good == []:
            total_filt_percent = 100  # all points were filtered
            total_correct = 0
        else:
            total_correct = (len(id_good[0])/ len(tot_match_pts[0])) *100 # 100 - ((len(id_good[0])/len(tot_match_pts[0])) * 100)
            total_filt_percent = 100 - total_correct

        table_cols = np.array((len(tot_match_pts[0]), (pts_notmatch/len(tot_match_pts[0])) * 100, tot_matched * 100, total_filt_percent, total_correct, (filt_magn/tot_m) * 100, (filt_dir/tot_m) * 100, (filt_snr/tot_m) * 100, (filt_memb/tot_m) * 100, (filt_patch/tot_m) * 100)).reshape(10, 1)
        result_postfilt_table = np.column_stack((result_postfilt_table, table_cols))


        ## Validation points filter

        # Check if validation points available and load validation points
        if val_pts_option != "":
            val_x1, val_y1, val_x2, val_y2 = np.loadtxt(val_pts_option, skiprows=1, unpack=True, delimiter="\t")

            # Transform world coordinates into pixel coordinates (in float)
            val_px1 = []; val_py1 = []; val_px2 = []; val_py2 = []
            for i in range(0, len(val_x1)):
                val_px1.append((xOrigin - val_x1[i]) / pixelHeight)
                val_py1.append((val_y1[i] - yOrigin) / pixelHeight)
                val_px2.append((xOrigin - val_x2[i]) / pixelHeight)
                val_py2.append((val_y2[i] - yOrigin) / pixelHeight)

            # Calculate the magnitude and angle of vectors between the validation points of both times
            val_magn = np.sqrt((np.array(val_px2) - np.array(val_px1)) ** 2 + (np.array(val_py2) - np.array(val_py1)) ** 2)
            val_angle = np.arctan2((np.array(val_py2) - np.array(val_py1)), (np.array(val_px2) - np.array(val_px1)))
            val_angle = np.degrees(val_angle) % 360

            magnitude_filt2D = np.empty(np.shape(magnitude)) * np.nan
            magnitude_filt2D[id_good2D] = magnitude[id_good2D]  # magnitude array (from matching) filtered

            alpha_filt2D = np.empty(np.shape(alpha)) * np.nan
            alpha_filt2D[id_good2D] = alpha[id_good2D]  # alpha array (from matching) filtered

            win_val_size = step_grid  # !!!advanced_param: That's the window size around the validation points
            magnitude_filt2D = np.pad(magnitude_filt2D, [(win_val_size,), (win_val_size,)], 'constant', constant_values=np.nan)
            alpha_filt2D = np.pad(alpha_filt2D, [(win_val_size,), (win_val_size,)], 'constant', constant_values=np.nan)

            match_magn = np.empty(len(val_magn))
            match_angle = np.empty(len(val_magn))

            # Correct validation pixel coordinates (point 'p') based on padding
            v_py1 = np.asarray(val_py1, np.int) + win_val_size; v_px1 = np.asarray(val_px1, np.int) + win_val_size

            for j in range(0, len(val_px1)):

                # Calculate mean matching magnitude at the validation point location. The mean is calculated within a window
                magn_win_atval = magnitude_filt2D[v_py1[j]-win_val_size : v_py1[j]+(win_val_size+1), v_px1[j]-win_val_size : v_px1[j]+(win_val_size+1)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    match_magn[j] = np.nanmedian(magn_win_atval)  # can be that there are only np.nans and return warning

                # Mean matching angle at the validation point location
                angle_win_atval = alpha_filt2D[v_py1[j]-win_val_size : v_py1[j]+(win_val_size+1), v_px1[j]-win_val_size : v_px1[j]+(win_val_size+1)]

                # Calculate mean angle of the window ('p' included)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    win_alpha_sin_mean = np.nanmean(np.sin(np.radians(angle_win_atval)))
                    win_alpha_cos_mean = np.nanmean(np.cos(np.radians(angle_win_atval)))
                    win_tan_mean = np.arctan2(win_alpha_sin_mean, win_alpha_cos_mean)  # in radians
                    match_angle[j] = np.degrees(win_tan_mean) % 360  # in angles (0-360°)
            # end loop

            # Calculate RMSE and R2 between the validation points and the magnitude obtained from matching
            ix = np.where(~np.isnan(match_magn))
            if len(ix[0]) >= 2:  # need at least to points to derive R2 (2 not optimal, value could be changed)
                RMSE_magn_val = (np.sqrt(np.nansum(((match_magn[ix] - val_magn[ix]) ** 2))/len(val_magn[ix]))) * pixelWidth
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(match_magn[ix], val_magn[ix])
                R2_magn_val = r_value

                angle_diff = ((match_angle[ix] - val_angle[ix]) + 180) % 360 - 180
                RMSE_angle_val = np.sqrt(np.nansum(((angle_diff) ** 2))/len(val_angle[ix]))
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(match_angle[ix], val_angle[ix])
                R2_angle_val = r_value

            else:
                RMSE_magn_val, R2_magn_val, RMSE_angle_val, R2_angle_val = np.nan, np.nan, np.nan, np.nan

            # Fill validation result table
            result_validation_table = np.column_stack((result_validation_table, match_magn, match_angle))

            # Fill validation result table sorted
            result_validation_table2 = np.row_stack((result_validation_table2, np.column_stack((RMSE_magn_val, RMSE_angle_val, R2_magn_val, R2_angle_val))))

        # end if val_pt_option


        # Calculate the remaining computational time for the post-filtering function
        end_time = time.time()
        if count_table == 1:  # print remaining time only afer the first loop
            est_time = (end_time - start_time) * (len(table_list) - 1)
            print("Estimated remaining time for post-filtering: ", str(datetime.timedelta(seconds=est_time)))


        # # If this section is uncommented and the 'res_plots' and 'mode' parameters added to the function parameters, the velocity fields for all individual combinations are created.
        # save_fig = res_plots + "\\" + str(file_names[count_table-1]) + ".jpeg"
        # if mode == 1 or mode == 3:
        #     input1 = res_plots + '\\I1_resize.tif'  # 2m resolution img
        #     img = mpimg.imread(input1)
        # if mode == 2:
        #     input1 = res_plots + '\\H1_resize.tif'  # 2m resolution img
        #     img = mpimg.imread(input1)
        #
        # with warnings.catch_warnings():  # get error message if the quiver plot gets np.nans to plot and when id_good is empty (id_good=[])
        #     warnings.simplefilter("ignore")
        #
        #     # --- Plot mean vector field (raw)
        #     fig = plt.figure(figsize=(18.0, 18.0))
        #     plt.ioff()
        #     plt.subplot(2, 2, 1)
        #     plt.imshow(img, cmap="gray", alpha=0.5, extent=[0, cols, 0, rows], origin='lower', aspect="auto")
        #     cmap = plt.cm.jet
        #     cmap.set_bad('white', 1.)
        #     qq = plt.quiver(x1, y1, d_x2, d_y2, (magn * pixelWidth), cmap=cmap, linewidth=3, angles='xy')
        #     plt.ylabel("Y-coordinates")
        #     plt.xlabel("X-coordinates")
        #     plt.title("Raw vectors field (m)")
        #     plt.gca().invert_yaxis()
        #
        #     # plot vector field (filtered)
        #     plt.subplot(2, 2, 2)
        #     cmap = plt.cm.jet
        #     cmap.set_bad('white', 1.)
        #     plt.imshow(img, cmap="gray", alpha=0.5, extent=[0, cols, 0, rows], origin='lower', aspect="auto")
        #     qq = plt.quiver(x1[id_good], y1[id_good], d_x2[id_good], d_y2[id_good], (magn[id_good] * pixelWidth), cmap=cmap, linewidth=3, angles='xy')
        #     plt.ylabel("Y-coordinates")
        #     plt.xlabel("X-coordinates")
        #     plt.title("Filtered vectors field (m)")
        #     plt.gca().invert_yaxis()
        #
        #     plt.subplot(2, 2, 3)
        #     plt.text(0.3, 0.5, ' ')
        #
        #     plt.subplot(2, 2, 4)
        #
        #     def discrete_cmap(N, base_cmap=None):
        #         """Create an N-bin discrete colormap from the specified input map"""
        #         base = plt.cm.get_cmap(base_cmap)
        #         color_list = base(np.linspace(0, 1, N))
        #         color_list[0] = [0, 0, 0, 0]
        #         cmap_name = base.name + str(N)
        #         return base.from_list(cmap_name, color_list, N)
        #
        #     plt.scatter(x1, y1, c=pts_filtered, s=10, cmap=discrete_cmap(48, 'jet'))
        #     plt.colorbar(ticks=range(48))
        #     plt.clim(0, 48)
        #     plt.gca().invert_yaxis()
        #     fig.savefig(save_fig, dpi=400)
        #     plt.clf()
        #     plt.close('all')

        count_table = count_table + 1  # counter for nbr of matching tables to filter

    # end of loop over matching tables


    ## Calculate the median and std for the magnitude as well as mean and std for the angle of all stacked matching tables (not for each table but for each point 'p')

    # Calculate median and standard deviation of all displacements
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_magn_stack = np.nanmedian(stack_magn, axis=1)  # stacking all combination per point 'p', so for each row
        std_magn_stack = np.nanstd(stack_magn, axis=1)

    # Calculate the number of values (members) per row  --> how many combination members per point 'p'
    nbr_stack_members = np.empty(np.shape(angle)) * np.nan
    for rr in range(0, len(y1)):
        nbr_stack_members[rr] = np.sum(~np.isnan(stack_magn[rr]))

    # Calculate mean and standard deviation of all angles
    mean_angle_stack = np.empty(np.shape(angle)) * np.nan
    std_angle_stack = np.empty(np.shape(angle)) * np.nan
    for i in range(0, len(y1)):
        angle_members = stack_angle[i, :]
        angle_members = angle_members[np.where(~np.isnan(angle_members))]  # angle_members can be empty [] because there are only np.nan, so when one point 'p' was filtered in every combination

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            win_alpha_sin_mean = np.nanmean(np.sin(np.radians(angle_members)))
            win_alpha_cos_mean = np.nanmean(np.cos(np.radians(angle_members)))
            win_tan_mean = np.arctan2(win_alpha_sin_mean, win_alpha_cos_mean)
            win_angle_mean = np.degrees(win_tan_mean) % 360

            mean_angle_stack[i] = win_angle_mean
            abs_diff_2 = np.abs(((angle_members - win_angle_mean) + 180) % 360 - 180) ** 2
            std_angle_stack[i] = np.sqrt((np.nansum(abs_diff_2)/np.size(angle_members)))
    # end of loop

    ## Remove outliers
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # can be np.nans
        p95 = np.nanpercentile(mean_magn_stack, 95)
        id_largestd = np.where(mean_magn_stack > p95)

    mean_magn_stack2 = mean_magn_stack  # make a copy of that array which will be changed
    mean_angle_stack2 = mean_angle_stack

    stack_magn2D = np.empty(np.shape(magnitude)) * np.nan
    stack_angle2D = np.empty(np.shape(alpha)) * np.nan
    for i in range(0, len(x1)):  # range(len(x1))
        stack_magn2D[y1[i], x1[i]] = mean_magn_stack[i]
        stack_angle2D[y1[i], x1[i]] = mean_angle_stack[i]


    ##--- MOVING WINDOWS ---
    # --- Initialization

    # Set moving window size
    wind_size = step_grid * times_stepgrid
    if wind_size % 2 == 0:
        wind_size = wind_size + 1  # needs to be odd to have a center
    hwind_size = int(wind_size / 2)

    # Padding arrays with np.nan
    stack_magn2D_pad = np.pad(stack_magn2D, [(hwind_size,), (hwind_size,)], 'constant', constant_values=np.nan)
    stack_angle2D_pad  = np.pad(stack_angle2D, [(hwind_size,), (hwind_size,)], 'constant', constant_values=np.nan)

    filt_magn = 0; filt_dir = 0
    id_largestd = id_largestd[0]

    # --- Start moving windows
    for i in id_largestd:  # make window on each matched point 'p'

        ## Correct y1 and x1 pixel coordinates (point 'p') based on padding
        py, px = y1[i] + hwind_size, x1[i] + hwind_size

        win_magn = stack_magn2D_pad[py - hwind_size: py + (hwind_size + 1), px - hwind_size: px + (hwind_size + 1)]
        win_alpha = stack_angle2D_pad [py - hwind_size: py + (hwind_size + 1), px - hwind_size: px + (hwind_size + 1)]

        # if np.nan at that point, pass, do not filter
        if np.nansum(stack_magn2D_pad[py, px]) == 0:
            mean_magn_stack2[i] = mean_magn_stack[i]
        else:
            win_magn = win_magn[np.where(~np.isnan(win_magn))]
            magn_neighbours = win_magn.flatten()
            win_alpha2 = win_alpha[np.where(~np.isnan(win_alpha))]
            alpha_neighbours = win_alpha2.flatten()

            ## Magnitude filter

            # Calculate the magnitude difference between point 'p' and its neighbours
            magn_max = 2 / pixelWidth  # !!!advanced_param: maximum percentage that the matched points can have with their neighbouring points
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                magn_diff = np.abs(magn_neighbours - stack_magn2D_pad[py, px])
                magn_idx = np.where(magn_diff <= magn_max)
                nbr_similar_magn_neighbours = magn_neighbours[magn_idx]

            # Calculate percentage of similar neighbours (neighbours that have a difference in magnitude smaller than magn_max)
            try:
                percent_sim_neighbours = len(nbr_similar_magn_neighbours) / len(magn_neighbours)
            except ZeroDivisionError:
                percent_sim_neighbours = 0

            std_nbr_magn = 3
            percent_neighbours = 0.5  # !!!advanced_param: if a point 'p' has less than 50% (0.5 = default) neighbours that have an angle differing of more than 'angle_max'.
            if np.abs(stack_magn2D_pad[py, px] - np.nanmedian(win_magn)) > (std_nbr_magn * np.nanstd(win_magn)) or percent_sim_neighbours <= percent_neighbours:
                # if point 'p' is filtered, replace with median value of window
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    win_minus_p = np.delete(win_magn, stack_magn2D_pad[py, px])
                    mean_magn_stack2[i] = np.nanmedian(win_minus_p)
                    stack_magn2D_pad[py, px] = np.nanmedian(win_minus_p)
                    filt_magn = filt_magn + 1
            else:
                # if point 'p' correct
                mean_magn_stack2[i] = mean_magn_stack[i]


            ## Directional filter

            # Select the non-nan values to calculate sd
            idy, idx = np.where(~np.isnan(win_alpha))
            values_angle_wind_all = win_alpha[idy, idx]
            std_nbr_dir = 3  # !!!advanced_param: number of standard deviation used for setting the filtering threshold for the directional filter

            # Calculate mean angle of the window ('p' included)
            win_alpha_sin_mean = np.nanmean(np.sin(np.radians(win_alpha)))
            win_alpha_cos_mean = np.nanmean(np.cos(np.radians(win_alpha)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                win_tan_mean = np.arctan2(win_alpha_sin_mean, win_alpha_cos_mean)  # in radians
                win_angle_mean = np.degrees(win_tan_mean) % 360  # in angles (0-360°)

                # Calculate std of the window ('p' included)
                abs_diff_2 = np.abs(((values_angle_wind_all - win_angle_mean) + 180) % 360 - 180) ** 2
                win_angle_std = np.sqrt(np.nansum(abs_diff_2) / np.size(values_angle_wind_all))

            # Calculate the angle difference between point 'p' and its neighbours
            angle_max = 20  # !!!advanced_param: maximum angle that the matched points can have with their neighbouring points
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                anglediff = np.abs(((alpha_neighbours - stack_angle2D_pad[py, px]) + 180) % 360 - 180)
                angle_idx = np.where(anglediff <= angle_max)
                nbr_similar_alpha_neighbours = alpha_neighbours[angle_idx]

            # Calculate percentage of similar neighbours (neighbours that have a difference in angle smaller than angle_max)
            try:
                percent_sim_neighbours = len(nbr_similar_alpha_neighbours) / len(alpha_neighbours)
            except ZeroDivisionError:
                percent_sim_neighbours = 0

            # Calculate the angle difference between point 'p' and the mean of the window
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                diff_angle_p_win = np.abs(((stack_angle2D_pad[py, px] - win_angle_mean) + 180) % 360 - 180)

            percent_neighbours = 0.5  # !!!advanced_param: if a point 'p' has less than 50% (0.5 = default) neighbours that have an angle differing of more than 'angle_max'.
            if diff_angle_p_win > (std_nbr_dir * win_angle_std) or percent_sim_neighbours <= percent_neighbours:
                # if point 'p' is filtered, go see in mean_magn_stack
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    win_minus_p = np.delete(win_alpha, stack_angle2D_pad[py, px])

                    # Calculate mean angle of the window ('p' included)
                    win_alpha_sin_mean = np.nanmean(np.sin(np.radians(win_minus_p)))
                    win_alpha_cos_mean = np.nanmean(np.cos(np.radians(win_minus_p)))

                    win_tan_mean = np.arctan2(win_alpha_sin_mean, win_alpha_cos_mean)  # in radians
                    win_angle_mean = np.degrees(win_tan_mean) % 360  # in angles (0-360°)

                    mean_angle_stack2[i] = win_angle_mean
                    stack_angle2D_pad[py, px] = win_angle_mean
                    filt_dir = filt_dir + 1

                # if point 'p' correct
                mean_angle_stack2[i] = mean_angle_stack[i]
    # end of moving window


    ## Get end product!

    # From mean magnitude and angle, retrieve mean x- and y-deviation
    new_x2 = x1 + np.cos(np.radians(mean_angle_stack2)) * mean_magn_stack
    new_y2 = y1 + np.sin(np.radians(mean_angle_stack2)) * mean_magn_stack
    stack_d_x2 = new_x2 - x1
    stack_d_y2 = new_y2 - y1

    mean_magn_stack2D = np.full((rows, cols), np.nan)
    for i in range(len(x1)):
        mean_magn_stack2D[y1[i], x1[i]] = mean_magn_stack2[i]  # distance of vector between p1 and p2 in the x direction (cols)

    idx1D = np.where(~np.isnan(mean_magn_stack2))
    idx2D = np.where(~np.isnan(mean_magn_stack2D))


    ##--- SAVE RESULTS IN TABLE ---
    # --- Save mean results (from stacked matching tables) ---
    table_mean_results = res_postfilt + '\\stack_results.txt'
    table_result_mean = np.column_stack((x1, y1, stack_d_x2, stack_d_y2, mean_magn_stack2, std_magn_stack, mean_angle_stack2, std_angle_stack, nbr_stack_members))

    # Headers
    line1 = 'This table has been created from the software IDMatch and contains the results for the mean and std of all displacement and angle combinations.'
    line2 = ''
    line3 = 'Variables:'
    line4 = 'x1, y1 = x- and y- the coordinates of a point matched (p) in image 1 (in pixel)'
    line5 = 'stack_d_x2, stack_d_y2 = pixel deviation in x and y direction'
    line6 = 'mean_magn = mean displacement of all combinations at point p'
    line7 = 'std_magn = std displacement of all combinations at point p'
    line8 = 'mean_angle = mean angle (in degrees) of all combinations at point p'
    line9 = 'std_angle = std angle (in degrees) of all combinations at point p'
    line10 = 'nbr_members = the number of members used to compute the mean and std'
    line11 = ''
    line12 = 'x1 y1 stack_d_x2 stack_d_y2 mean_magn std_magn mean_angle std_angle nbr_members'
    header_matchtable = '\n'.join([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12])
    np.savetxt(table_mean_results, table_result_mean, fmt='%1.4f', newline='\n', header=header_matchtable, delimiter='\t', comments='# ', encoding=None)

    ##--- Save results from post-filtering (% of points filtered per filter) ---
    # Save results of percentage of matching points discarded from the different post-filters
    table_postfilt = res_path + "\\table_results_postfilt.txt"

    # Headers
    line1 = 'This table has been created from the software IDMatch and contains the results from the post-processing function.'
    line2 = 'Each image/dsm/hillshade pair matching results went through a post-filtering procedure containing several filters.'
    line3 = 'The number of matched points and filtered points per filter (in percent) is displayed for each combination. The percentage values do not sum up to 100% as one point p may have been filtered by several different filters.'
    line4 = ''
    line5 = 'The first value is the number of total points p used for matching.'
    line6 = 'The second value is the percentage of points that were not matched.'
    line7 = 'The third value is the percentage of points that were matched. The percentage values below are based on on the total MATCHED points.'
    line8 = 'The fourth value is the percentage of points that were filtered out by the post-filtering procedure.'
    line9 = 'The fifth value is the percentage of points that are correct (not filtered out).'
    line10 = 'The sixth value is the percentate of points that were filtered with the magnitude filter.'
    line11 = 'The seventh value is the percentate of points that were filtered with the directional filter.'
    line12 = 'The eighth value is the percentate of points that were filtered with the SNratio filter (if matching method M3 chosen, it returns np.nan).'
    line13 = 'The nineth value is the percentage of points that did not go through filtering due to a loo low nbr of members in window.'
    line14 = 'The tenth value is the percentage of points that were discarded because the percentage of total remaining points after the post-filtering procedure was below a certain threshold.'
    line15 = ''
    line16 = '\t'.join(file_names)
    header_resulttable = '\n'.join([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13, line14, line15, line16])
    np.savetxt(table_postfilt, result_postfilt_table, fmt='%1.2f', delimiter='\t', newline='\n', header=header_resulttable)



    # If validation points available, validate the stacked velocity field against validation points
    if val_pts_option != "":

        # Validate points on correct averaged velocity field

        # Use the 2D arrays of above, take out hwind_size padding
        stack_magn2D = stack_magn2D_pad[hwind_size:-hwind_size, hwind_size:-hwind_size]
        stack_angle2D = stack_angle2D_pad[hwind_size:-hwind_size, hwind_size:-hwind_size]

        # Pad
        win_val_size = step_grid  # !!!advanced_param: That's the window size around the validation points
        magn_stack2D_pad = np.pad(stack_magn2D, [(win_val_size,), (win_val_size,)], 'constant', constant_values=np.nan)
        angle_stack2D_pad = np.pad(stack_angle2D, [(win_val_size,), (win_val_size,)], 'constant', constant_values=np.nan)

        match_magn = np.empty(len(val_magn))
        match_angle = np.empty(len(val_magn))

        for j in range(0, len(val_px1)):
            # Calculate mean matching magnitude at the validation point location. The mean is calculated within a window
            magn_win_atval = magn_stack2D_pad[(v_py1[j] - win_val_size): (v_py1[j] + (win_val_size + 1)), (v_px1[j] - win_val_size): (v_px1[j] + (win_val_size + 1))]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                match_magn[j] = np.nanmedian(magn_win_atval)  # can be that there are only np.nans and return warning

            angle_win_atval = angle_stack2D_pad[v_py1[j] - win_val_size: v_py1[j] + (win_val_size + 1), v_px1[j] - win_val_size: v_px1[j] + (win_val_size + 1)]
            # Calculate mean matching angle at the validation point location
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                win_alpha_sin_mean = np.nanmean(np.sin(np.radians(angle_win_atval)))
                win_alpha_cos_mean = np.nanmean(np.cos(np.radians(angle_win_atval)))
                win_tan_mean = np.arctan2(win_alpha_sin_mean, win_alpha_cos_mean)  # in radians
                match_angle[j] = np.degrees(win_tan_mean) % 360  # in angles (0-360°)
        # end loop

        # Calculate RMSE and R2 between the validation points and the stacked magnitude and angle
        ix = np.where(~np.isnan(match_magn))

        if np.size(ix) == 0:  # possible is there are no point 'p' from the stacked mean around the validation point (can be in case of a small number of iteration)
            RMSE_magn_val, R2_magn_val, RMSE_angle_val, R2_angle_val  = np.nan,  np.nan,  np.nan,  np.nan
        else:
            RMSE_magn_val = np.sqrt(np.nansum(((match_magn[ix] - val_magn[ix]) ** 2)) / len(val_magn[ix]))
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(match_magn[ix], val_magn[ix])
            R2_magn_val = r_value


            angle_diff = ((match_angle[ix] - val_angle[ix]) + 180) % 360 - 180
            RMSE_angle_val = np.sqrt(np.nansum(((angle_diff) ** 2)) / len(val_angle[ix]))
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(match_angle[ix], val_angle[ix])
            R2_angle_val = r_value

        # Fill validation result table
        result_validation_table = np.column_stack((result_validation_table, match_magn, match_angle))

        # Fill validation result sorted combi
        result_validation_table2 = np.row_stack((result_validation_table2, np.column_stack((RMSE_magn_val, RMSE_angle_val, R2_magn_val, R2_angle_val))))


        ##--- Save validation points for all tables ---
        table_valid_pts = res_path + '\\validation_pts.txt'
        table_validation = np.column_stack((val_px1, val_py1, val_px2, val_py2, val_magn, val_angle, result_validation_table))
        file_names_m = [s + '_m' for s in file_names]
        file_names_a = [s + '_a' for s in file_names]
        file_name_all = [i for i in zip(file_names_m, file_names_a)]
        file_name_all2 = [item for sublist in file_name_all for item in sublist]
        file_name_all = file_name_all2 + ['stack_m'] + ['stack_a']

        # Headers
        line1 = 'This table has been created from the software IDMatch and contains the original validation points given by the user along other information'
        line2 = ''
        line3 = 'Variables:'
        line4 = 'val_px1, val_py1 = x- and y- the coordinates of the validation points in the first image (in pixel)'
        line5 = 'val_px2, val_py2 = x- and y- the coordinates of the validation points in the second image (in pixel)'
        line6 = 'val_magn = The the displacement (magnitude) between the validation points in image1 and image 2 (in pixel)'
        line7 = 'val_angle = The the angle between the validation points in image1 and image 2 (in degrees)'
        line8 = 'xfilname_m = The displacement (magnitude) at the location of the validation points, calculated from a window on the matching displacement (calculated at every grid_step, in pixel)'
        line9 = 'xfilname_a = The angle (in degree) at the location of the validation points, calculated from a window on the matching angles (calculated at every grid_step, in pixel)'
        line10 = ''
        line11 = ''
        line12 = ' '.join(['val_px1 val_py1 val_px2 val_py2 val_magn val_angle'] + file_name_all)
        header_matchtable = '\n'.join( [line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12])
        np.savetxt(table_valid_pts, table_validation, fmt='%1.4f', newline='\n', header=header_matchtable, delimiter='\t', comments='# ', encoding=None)

        rmse_magn = result_validation_table2[:, 0]  # rmse magn
        r2_magn = result_validation_table2[:, 2]  # r2 magn
        rmse_angle = result_validation_table2[:, 1]  # rmse angle
        r2_angle = result_validation_table2[:, 3]  # r2 angle

        file_names_m = np.array(file_names_m + ['stack_m'])
        file_names_a = np.array(file_names_a + ['stack_a'])

        # Sort RMSE magn
        ind_nonnan = np.where(~np.isnan(rmse_magn))
        rmse_magn2 = rmse_magn[ind_nonnan]
        col_rmse_m = file_names_m[ind_nonnan]
        id_rmse_m = np.argsort(rmse_magn2, axis=0)
        id_rmse_m[np.argsort(rmse_magn2)] = np.arange(len(rmse_magn2))

        rmse_magn_s = np.zeros(len(id_rmse_m))
        col_rmse_m_s = [None] * (len(id_rmse_m))
        for i in range(0, len(id_rmse_m)):
            ix = np.where(id_rmse_m == i)
            rmse_magn_s[i] = rmse_magn2[ix[0]]
            col_rmse_m_s[i] = col_rmse_m[ix[0]]

        # Sort RMSE angle
        ind_nonnan = np.where(~np.isnan(rmse_angle))
        rmse_angle2 = rmse_angle[ind_nonnan]
        col_rmse_a = file_names_a[ind_nonnan]
        id_rmse_a = np.argsort(rmse_angle2)
        id_rmse_a[np.argsort(rmse_angle2)] = np.arange(len(rmse_angle2))

        rmse_angle_s = np.zeros(len(id_rmse_a))
        col_rmse_a_s = [None] * (len(id_rmse_a))
        for i in range(0, len(id_rmse_a)):
            ix = np.where(id_rmse_a == i)
            rmse_angle_s[i] = rmse_angle2[ix[0]]
            col_rmse_a_s[i] = col_rmse_a[ix[0]]

        # Sort R2 magn
        ind_nonnan = np.where(~np.isnan(r2_magn))
        r2_magn2 = r2_magn[ind_nonnan]
        col_r2_m = file_names_m[ind_nonnan]
        id_r2_m = np.argsort(r2_magn2)
        id_r2_m[np.argsort(r2_magn2)] = np.arange(len(r2_magn2))

        r2_magn_s = np.zeros(len(id_r2_m))
        col_r2_m_s = [None] * (len(id_r2_m))
        for i in range(0, len(id_rmse_m)):
            ix = np.where(id_r2_m == i)
            r2_magn_s[i] = r2_magn2[ix[0]]
            col_r2_m_s[i] = col_r2_m[ix[0]]

        # Sort R2 angle
        ind_nonnan = np.where(~np.isnan(r2_angle))
        r2_angle2 = r2_angle[ind_nonnan]
        col_r2_a = file_names_a[ind_nonnan]
        id_r2_a = np.argsort(r2_angle2)
        id_r2_a[np.argsort(r2_angle2)] = np.arange(len(r2_angle2))

        r2_angle_s = np.zeros(len(id_r2_a))
        col_r2_a_s = [None] * (len(id_r2_a))
        for i in range(0, len(id_r2_a)):
            ix = np.where(id_r2_a == i)
            r2_angle_s[i] = r2_angle2[ix[0]]
            col_r2_a_s[i] = col_r2_a[ix[0]]

        result_sorted_table = np.column_stack((rmse_magn_s.astype(str), col_rmse_m_s, rmse_angle_s.astype(str), col_rmse_a_s, np.flip(r2_magn_s, axis=0).astype(str), np.flip(col_r2_m_s, axis=0), np.flip(r2_angle_s, axis=0).astype(str), np.flip(col_r2_a_s, axis=0)))

        print("Based on the validation points, the combination of velocity field ...")
        print(" with the smallest RMSE (in meter) for the displacement is: ", result_sorted_table[0][1], " (RMSE = ", result_sorted_table[0][0], ")")
        print(" with the smallest coefficient of determination (r2) is: ", result_sorted_table[0][5], " (r2 = ", result_sorted_table[0][4], ")")

        ##--- Save validation points for all tables ---
        table_sort_combi = res_path + '\\sorted_best_combi.txt'

        # Headers
        line1 = 'This table has been created from the software IDMatch and contains the sorted RMSE and R2 calculated between the matched and the validation points for all combinations'
        line2 = ''
        line3 = 'Variables:'
        line4 = 'RMSE = Root mean Square Error'
        line5 = 'R2 = Coefficient of determination of a linear regression'
        line6 = ''
        line7 = 'the _m and _a after each dataset means "magnitude" or "angle"'
        line8 = ''
        line9 = '\t'.join(["RMSE_m_values RMSE_m_name RMSE_a_values RMSE_a_name R2_m_values R2_m_name R2_a_values R2_a_name"], )
        header_matchtable = '\n'.join([line1, line2, line3, line4, line5, line6, line7, line8, line9])
        np.savetxt(table_sort_combi, result_sorted_table, fmt="%s", newline='\n', header=header_matchtable, delimiter='\t', comments='# ', encoding=None)

    else:
        table_valid_pts = ""
        table_sort_combi = ""

    # end if val_pt_option

    return table_mean_results, table_postfilt, table_valid_pts, table_sort_combi



def display_results(res_postfilt, res_plots, table_mean_results, data_info, table_postfilt, mode, table_valid_pts, val_pts_option):
    """
       This function plots the velocity fields and other figures.

          Parameters
          ----------

          res_path : str
               Path to the result folder that has been defined by the user in the params.py file.


          Returns
          -------

          """

    ## Message:
    print("## Generating report")

    ## --- 1. Plotting ---
    dpi_value = 400  # choose resolution of plots

    # Import the mean matching result table
    x1, y1, d_x2, d_y2, mean_magn, std_magn, mean_angle, std_angle, nbr_members = np.loadtxt(table_mean_results, skiprows=12, unpack=True, delimiter='\t')
    x1 = x1.astype(int); y1 = y1.astype(int)

    # Get infos on the dataset size
    xOrigin, pixelWidth, yOrigin, pixelHeight, rows, cols, proj = data_info

    # Re-create vx, vy, length, angle and snr 2D arrays from table
    vx = np.full((rows, cols), np.nan)
    vy = np.full((rows, cols), np.nan)
    mean_displ = np.full((rows, cols), np.nan)
    std_displ = np.full((rows, cols), np.nan)
    mean_alpha = np.full((rows, cols), np.nan)
    std_alpha = np.full((rows, cols), np.nan)
    nbr_member = np.full((rows, cols), np.nan)
    for i in range(len(x1)):
        vx[y1[i], x1[i]] = d_x2[i]  # distance of vector between p1 and p2 in the x direction (cols)
        vy[y1[i], x1[i]] = d_y2[i]  # distance of vector between p1 and p2 in the y direction (rows)
        mean_displ[y1[i], x1[i]] = mean_magn[i]
        std_displ[y1[i], x1[i]] = std_magn[i]
        mean_alpha[y1[i], x1[i]] = mean_angle[i]
        std_alpha[y1[i], x1[i]] = std_angle[i]
        nbr_member[y1[i], x1[i]] = nbr_members[i]  # number of members from all matching tables used to calculate mean and std

    # Import images or hillshade for plotting
    if mode == 1 or mode == 3:
        img = mpimg.imread(res_plots + '\\I1_resize.tif')
    if mode == 2:
        img = mpimg.imread(res_plots + '\\H1_resize.tif')

    ##--- PLOT THE MEAN RESULTS ---

    # --- Plot mean vector field (filtered)
    id_keep = np.where(~np.isnan(mean_magn))
    nbr_stack_pts = len(id_keep[0])
    print("The total points of the grid (points 'p') = ", len(x1), " and the number of matched points for the stacked velocity field = ", len(id_keep[0]))

    print("plot 1")
    fig_vect_field_filt = res_plots + '\\fig_avf_vector.jpeg'
    fig, ax = plt.subplots()
    plt.imshow(img, cmap="gray", alpha=0.5, extent=[0, cols, 0, rows], origin='lower')

    qq = ax.quiver(x1[id_keep], y1[id_keep], d_x2[id_keep], d_y2[id_keep], (mean_magn[id_keep]*pixelWidth), cmap=plt.cm.jet, linewidth=3, angles='xy')
    cbar = plt.colorbar(qq)
    ax.set_ylim(ax.get_ylim()[::-1])
    cbar.ax.set_ylabel('Displacement (m)')
    plt.ylabel("pixels")
    plt.xlabel("pixels")
    plt.title("Averaged velocity field (m)")
    fig.savefig(fig_vect_field_filt, bbox_inches='tight', dpi=dpi_value)
    plt.clf()

    ##---SCATTER PLOTS
    # --- Plot AVF with magnitude as points
    print("plot 2")
    fig_mean_magn_scatter = res_plots + '\\fig_avf_points.jpeg'
    fig, ax = plt.subplots()
    plt.scatter(x1, y1, c= mean_magn*pixelWidth, s=10)
    plt.jet()
    plt.colorbar()
    plt.ylabel("pixels")
    plt.xlabel("pixels")
    plt.title("Averaged velocity field (m)")
    plt.gca().invert_yaxis()
    fig.savefig(fig_mean_magn_scatter, bbox_inches='tight', dpi=dpi_value)
    plt.clf()

    # --- Plot standard deviation fo the magnitude for AVF
    print("plot 3")
    fig_std_magn_scatter = res_plots + '\\fig_avf_std_points.jpeg'
    fig, ax = plt.subplots()
    plt.scatter(x1, y1, c=std_magn*pixelWidth, s=10)
    plt.jet()
    plt.colorbar()
    plt.ylabel("pixels")
    plt.xlabel("pixels")
    plt.title("Standard deviation of the averaged velocity field (m)")
    plt.gca().invert_yaxis()
    fig.savefig(fig_std_magn_scatter, bbox_inches='tight', dpi=dpi_value)
    plt.clf()

    ## --- Plot number of combinations per point p
    print("plot 4")
    fig_nbr_members_meanstd = res_plots + '\\fig_nbr_combi_perpoint.jpeg'
    fig, ax = plt.subplots()
    plt.scatter(x1, y1, c=nbr_members, s=5)
    plt.jet()
    plt.colorbar()
    plt.ylabel("pixels")
    plt.xlabel("pixels")
    plt.title("Number of combination per point 'p'")
    plt.gca().invert_yaxis()
    fig.savefig(fig_nbr_members_meanstd, bbox_inches='tight', dpi=dpi_value)
    plt.clf()


    ## --- PLOT HISTOGRAMS

    ## --- Plot histogram avf
    print("plot 5")
    fig_hist_mean_magn = res_plots + '\\fig_avf_histogram.jpeg'

    fig, ax = plt.subplots()
    plt.hist((mean_magn[id_keep]*pixelWidth), bins=50)
    plt.ylabel("Frequency of occurence")
    plt.xlabel("Bins (displacement in meters)")
    plt.title("Histogram of the averaged velocity field")
    fig.savefig(fig_hist_mean_magn, bbox_inches='tight', dpi=dpi_value)
    plt.clf()

    ## --- Plot histogram std magnitude
    print("plot 6")
    fig_hist_std_magn = res_plots + '\\fig_avf_std_histogram.jpeg'
    fig, ax = plt.subplots()
    plt.hist((std_magn[id_keep] * pixelWidth), bins=50)
    plt.ylabel("Frequency of occurence")
    plt.xlabel("Bins (displacement in meters)")
    plt.title("Histogram of the standard deviation at all points p for the averaged velocity field")
    fig.savefig(fig_hist_std_magn, bbox_inches='tight', dpi=dpi_value)
    plt.clf()


    ## --- PLOT VALIDATION
    if val_pts_option != "":

        # ----- Import validation points table
        table_validation = np.loadtxt(table_valid_pts, skiprows=12, unpack=True, delimiter='\t')
        val_px1 = table_validation[0]
        val_py1 = table_validation[1]
        val_magn = table_validation[4]
        stack_magn_av = table_validation[-2]

        ## --- Plot scatter velocities vs validation velocities pts (displacement)
        print("plot 7")
        fig_scatter_magn = res_plots + '\\fig_validation_allcombi.jpeg'

        fig, ax = plt.subplots()
        for i in range(6, len(table_validation)-2):
            if i % 2 == 0:  # magnitude
                plt.scatter(table_validation[i] * pixelWidth, val_magn * pixelWidth, s=5)

        plt.scatter(stack_magn_av * pixelWidth, val_magn * pixelWidth, s=5, color="black")
        ax.set_xlim((0, np.nanmax(val_magn) * pixelWidth))
        ax.set_ylim((0, np.nanmax(val_magn) * pixelWidth))
        plt.ylabel("Displacement from matching (m)")
        plt.xlabel("Displacement from validation points (m)")
        plt.title("Scatter plot of the displacement")
        fig.savefig(fig_scatter_magn, bbox_inches='tight', dpi=dpi_value)
        plt.clf()

        # --- Plot 2D scatter plot of magnitude differences between AVF and validation points
        print("plot 8")
        fig_diff_magn_scatter = res_plots + '\\fig_differences_displacement.jpeg'

        diff_magn = np.abs(stack_magn_av - val_magn) * pixelWidth
        fig, ax = plt.subplots()
        plt.scatter(val_px1, val_py1, c=diff_magn, s=10)
        plt.jet()
        plt.colorbar()
        plt.ylabel("pixel")
        plt.xlabel("pixel")
        plt.title("Differences in displacements between the AVF and validation points(m)")
        plt.gca().invert_yaxis()
        fig.savefig(fig_diff_magn_scatter, bbox_inches='tight', dpi=400)
        plt.clf()

    #end loop validation pts

    ##--- Plot results for al combinations ---

    # Import all post-filtering tables (for the names for the validation and plotting all matching results later)
    tables_match = os.listdir(res_postfilt);
    tables_match_list = [];
    table_names = []
    for table in tables_match:
        if table.endswith("txt"):
            tables_match_list.append(os.path.join(res_postfilt, table))
            table_names.append(os.path.splitext(table)[0])
        else:
            raise Exception('The imported filed need to be .txt')
    table_names.sort(); tables_match_list.sort()
    table_names = table_names[:-1]  # take out the stack_result.txt
    tables_match_list = tables_match_list[:-1]

    # Plot the influence of matching methods and filters in 2D
    print("plot 9")

    info_filters = [[] for x in range(len(tables_match_list))]
    for j in range(0, len(tables_match_list)):
        storing_f = [None] * len(x1)

        if 'M1' in table_names[j]:
            x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
            p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
            p_filt = p_filt[0]
            for k in range(0, len(p_filt)):
                storing_f[p_filt[k]] = 'M1'

        elif 'M2' in table_names[j]:
            x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
            p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
            p_filt = p_filt[0]
            for k in range(0, len(p_filt)):
                storing_f[p_filt[k]] = 'M2'

        else:
            # 'M3' in table_names[j]:
            x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
            p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
            p_filt = p_filt[0]
            for k in range(0, len(p_filt)):
                storing_f[p_filt[k]] = 'M3'

        info_filters[j] = (storing_f)

    fig_all_combi_M = res_plots + '\\fig_all_combi_M.jpeg'
    fig, ax = plt.subplots()

    for i in range(0, len(x1)):
        lst2 = [item[i] for item in info_filters]
        lst2 = [x for x in lst2 if x is not None]

        count = 0
        if sum(['M1' in mystring for mystring in lst2]) > 0:
            count = count + 1
        if sum(['M2' in mystring for mystring in lst2]) > 0:
            count = count + 2
        if sum(['M3' in mystring for mystring in lst2]) > 0:
            count = count + 4

        if count == 1:
            plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.5, c='yellow')
        if count == 2:
            plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.5, c='red')
        if count == 3:
            plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.5, c='orange')
        if count == 4:
            plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.5, c='blue')
        if count == 5:
            plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.5, c='green')
        if count == 6:
            plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.5, c='violet')
        if count == 7:
            plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.5, c='black')

    plt.ylabel("Y-coordinates")
    plt.xlabel("X-coordinates")
    plt.title("Matched points per matching method")
    plt.gca().invert_yaxis()
    custom_lines = [Line2D([0], [0], color=('yellow'), lw=4, marker='o', linestyle=''),
                    Line2D([0], [0], color=('red'), lw=4, marker='o', linestyle=''),
                    Line2D([0], [0], color=('orange'), lw=4, marker='o', linestyle=''),
                    Line2D([0], [0], color=('blue'), lw=4, marker='o', linestyle=''),
                    Line2D([0], [0], color=('green'), lw=4, marker='o', linestyle=''),
                    Line2D([0], [0], color=('violet'), lw=4, marker='o', linestyle=''),
                    Line2D([0], [0], color=('black'), lw=4, marker='o', linestyle='')]
    ax.legend(custom_lines, ["M1", "M2", "M1M2", "M3", "M1M3", "M2M3", "M1M2M3"])
    fig.savefig(fig_all_combi_M, bbox_inches='tight', dpi=dpi_value)
    plt.clf()
    plt.close('all')

    if mode == 1:
        info_filters = [[] for x in range(len(tables_match_list))]
        for j in range(0, len(tables_match_list)):
            storing_f = [None] * len(x1)

            if 'F1' in table_names[j] and 'F2' in table_names[j]:
                x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
                p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
                p_filt = p_filt[0]
                for k in range(0, len(p_filt)):
                    storing_f[p_filt[k]] = 'F1 F2'

            elif 'F1' in table_names[j] and 'F3' in table_names[j]:
                x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
                p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
                p_filt = p_filt[0]
                for k in range(0, len(p_filt)):
                    storing_f[p_filt[k]] = 'F1 F3'

            else:
                if 'F1' in table_names[j]:
                    x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
                    p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
                    # info_filters[p_filt, j] = np.nansum([info_filters[p_filt, j], 1])
                    p_filt = p_filt[0]
                    for k in range(0, len(p_filt)):
                        storing_f[p_filt[k]] = 'F1'

                if 'F2' in table_names[j]:
                    x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
                    p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
                    # info_filters[p_filt, j] = np.nansum([info_filters[p_filt, j], 2])
                    p_filt = p_filt[0]
                    for k in range(0, len(p_filt)):
                        storing_f[p_filt[k]] = 'F2'

                if 'F3' in table_names[j]:
                    x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
                    p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
                    # info_filters[p_filt, j] = np.nansum([info_filters[p_filt, j], 4])
                    p_filt = p_filt[0]
                    for k in range(0, len(p_filt)):
                        storing_f[p_filt[k]] = 'F3'

                if 'F' not in table_names[j]:
                    x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
                    p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
                    # info_filters[p_filt, j] = np.nansum([info_filters[p_filt, j], 4])
                    p_filt = p_filt[0]
                    for k in range(0, len(p_filt)):
                        storing_f[p_filt[k]] = 'NF'

            info_filters[j] = storing_f

    elif mode == 2:
        info_filters = [[] for x in range(len(tables_match_list))]
        for j in range(0, len(tables_match_list)):
            storing_f = [None] * len(x1)

            if 'F1' in table_names[j] and 'F4' in table_names[j]:
                x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(
                    tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
                p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
                p_filt = p_filt[0]
                for k in range(0, len(p_filt)):
                    storing_f[p_filt[k]] = 'F1 F4'

            elif 'F1' in table_names[j] and 'F5' in table_names[j]:
                x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(
                    tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
                p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
                p_filt = p_filt[0]
                for k in range(0, len(p_filt)):
                    storing_f[p_filt[k]] = 'F1 F5'

            else:
                if 'F1' in table_names[j]:
                    x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(
                        tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
                    p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
                    p_filt = p_filt[0]
                    for k in range(0, len(p_filt)):
                        storing_f[p_filt[k]] = 'F1'

                if 'F4' in table_names[j]:
                    x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(
                        tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
                    p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
                    p_filt = p_filt[0]
                    for k in range(0, len(p_filt)):
                        storing_f[p_filt[k]] = 'F4'

                if 'F5' in table_names[j]:
                    x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
                    p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
                    p_filt = p_filt[0]
                    for k in range(0, len(p_filt)):
                        storing_f[p_filt[k]] = 'F5'

                if 'F' not in table_names[j]:
                    x1_m, y1_m, dx2_m, dy2_m, peakCorr_m, meanAbsCorr_m, SNratio_m, magn_m, angle_m, pts_filtered_m = np.loadtxt(tables_match_list[j], skiprows=14, unpack=True, delimiter='\t')
                    p_filt = np.where(pts_filtered_m == 0)  # take only the correct values
                    p_filt = p_filt[0]
                    for k in range(0, len(p_filt)):
                        storing_f[p_filt[k]] = 'NF'

            info_filters[j] = storing_f

    print("plot 10")
    if mode == 1:
        fig_all_combi_F = res_plots + '\\fig_all_combi_F.jpeg'
        fig, ax = plt.subplots()

        countF1 = 0; countF2 = 0; countF3 = 0; tot_no_filt = 0; tot_filt = 0; tot_no_match = 0; tot_both = 0
        # Plot the points with filers in colors
        for i in range(0, len(x1_m)):
            lst2 = [item[i] for item in info_filters]  # takes all points 'p' for all combinations
            lst2 = [x for x in lst2 if x is not None]

            # test only for with and without filter
            count = 0; countNoF = 0
            if sum(['F1' in mystring for mystring in lst2]) > 0:
                count = count + 1
                countF1 = countF1 + 1
            if sum(['F2' in mystring for mystring in lst2]) > 0:
                count = count + 1
                countF2 = countF2 + 1
            if sum(['F3' in mystring for mystring in lst2]) > 0:
                count = count + 1
                countF3 = countF3 + 1
            if sum(['NF' in mystring for mystring in lst2]) > 0:
                countNoF = countNoF + 1

            if count == 0 and countNoF == 0:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='white')
                tot_no_match = tot_no_match + 1
            elif count != 0 and countNoF == 0:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='orange')  # filter only
                tot_filt = tot_filt + 1
            elif count == 0 and countNoF != 0:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='blue')  # without filter only
                tot_no_filt = tot_no_filt + 1
            elif count != 0 and countNoF != 0:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='black')  # with both only
                tot_both = tot_both + 1


        plt.ylabel("pixel")
        plt.xlabel("pixel")
        plt.title("Matched points per filter")
        plt.gca().invert_yaxis()
        custom_lines = [Line2D([0], [0], color=('orange'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('blue'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('black'), lw=4, marker='o', linestyle='')]
        ax.legend(custom_lines, ["with pre-filter", "without pre-filter", "with + without pre-filter"])  # , "without_f"
        fig.savefig(fig_all_combi_F, bbox_inches='tight', dpi=dpi_value)
        plt.clf()

    elif mode == 2:

        fig_all_combi_F = res_plots + '\\fig_all_combi_F_withwithout.jpeg'
        fig, ax = plt.subplots()

        countF1 = 0; countF4 = 0; countF5 = 0; tot_no_filt = 0; tot_filt = 0; tot_no_match = 0; tot_both = 0
        # Plot the points with filers in colors
        for i in range(0, len(x1_m)):
            lst2 = [item[i] for item in info_filters]  # takes all points 'p' for all combinations
            lst2 = [x for x in lst2 if x is not None]

            count = 0; countNoF = 0
            if sum(['F1' in mystring for mystring in lst2]) > 0:
                count = count + 1
                countF1 = countF1 + 1
            if sum(['F4' in mystring for mystring in lst2]) > 0:
                count = count + 1
                countF4 = countF4 + 1
            if sum(['F5' in mystring for mystring in lst2]) > 0:
                count = count + 1
                countF5 = countF5 + 1
            if sum(['NF' in mystring for mystring in lst2]) > 0:
                countNoF = countNoF + 1

            if count == 0 and countNoF == 0:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='white')
                tot_no_match = tot_no_match + 1
            elif count != 0 and countNoF == 0:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='orange')  # filter only
                tot_filt = tot_filt + 1
            elif count == 0 and countNoF != 0:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='blue')  # without filter only
                tot_no_filt = tot_no_filt + 1
            elif count != 0 and countNoF != 0:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='black')  # with both only
                tot_both = tot_both + 1

        plt.ylabel("pixel")
        plt.xlabel("pixel")
        plt.title("Matched points per filter")
        plt.gca().invert_yaxis()
        custom_lines = [Line2D([0], [0], color=('orange'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('blue'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('black'), lw=4, marker='o', linestyle='')]
        ax.legend(custom_lines, ["with pre-filter", "without pre-filter", "with + without pre-filter"])  # , "without_f"
        fig.savefig(fig_all_combi_F, bbox_inches='tight', dpi=dpi_value)
        plt.clf()


    print("plot 11")
    if mode == 1:
        fig_all_combi_F = res_plots + '\\fig_all_combi_F.jpeg'
        fig, ax = plt.subplots()

        countF1 = 0; countF2 = 0; countF3 = 0; tot_no_filt = 0; tot_filt = 0; tot_no_match = 0; tot_both = 0
        # Plot the points with filers in colors
        for i in range(0, len(x1_m)):
            lst2 = [item[i] for item in info_filters]  # takes all points 'p' for all combinations
            lst2 = [x for x in lst2 if x is not None]
            count = 0
            if sum(['F1' in mystring for mystring in lst2]) > 0:
                count = count + 1
                countF1 = countF1 + 1
            if sum(['F2' in mystring for mystring in lst2]) > 0:
                count = count + 2
                countF2 = countF2 + 1
            if sum(['F3' in mystring for mystring in lst2]) > 0:
                count = count + 4
                countF3 = countF3 + 1
            if sum(['NF' in mystring for mystring in lst2]) > 0:
                countNoF = countNoF + 1


            if count == 1:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='yellow')
            elif count == 2:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='red')
            elif count == 3:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='orange')
            elif count == 4:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='blue')
            elif count == 5:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='green')
            elif count == 6:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.5, c='violet')
            elif count == 7:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='black')
            elif count == 0:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='white')


        plt.ylabel("pixels")
        plt.xlabel("pixels")
        plt.title("Matched points per filter")
        plt.gca().invert_yaxis()
        custom_lines = [Line2D([0], [0], color=('yellow'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('orange'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('red'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('blue'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('green'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('violet'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('black'), lw=4, marker='o', linestyle='')]
        ax.legend(custom_lines, ["F1", "F2", "F1F2", "F3", "F1F3", "F2F3", "F1F2F3"])
        fig.savefig(fig_all_combi_F, bbox_inches='tight', dpi=dpi_value)
        plt.clf()

    elif mode == 2:

        fig_all_combi_F = res_plots + '\\fig_all_combi_F_withwithout.jpeg'
        fig, ax = plt.subplots()

        countF1 = 0; countF4 = 0; countF5 = 0; tot_no_filt = 0; tot_filt = 0; tot_no_match = 0; tot_both = 0
        # Plot the points with filers in colors
        for i in range(0, len(x1_m)):
            lst2 = [item[i] for item in info_filters]  # takes all points 'p' for all combinations
            lst2 = [x for x in lst2 if x is not None]

            count = 0
            if sum(['F1' in mystring for mystring in lst2]) > 0:
                count = count + 1
                countF1 = countF1 + 1
            if sum(['F4' in mystring for mystring in lst2]) > 0:
                count = count + 2
                countF4 = countF4 + 1
            if sum(['F5' in mystring for mystring in lst2]) > 0:
                count = count + 4
                countF5 = countF5 + 1
            if sum(['NF' in mystring for mystring in lst2]) > 0:
                countNoF = countNoF + 1

            if count == 1:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='yellow')
            if count == 2:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='red')
            if count == 3:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='orange')
            if count == 4:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='blue')
            if count == 5:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='green')
            if count == 6:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.5, c='violet')
            if count == 7:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='black')
            elif count == 0:
                plt.scatter(x1_m[i].astype(int), y1_m[i].astype(int), s=0.3, c='white')

        plt.ylabel("Y-coordinates")
        plt.xlabel("X-coordinates")
        plt.title("Matched points per filter")
        plt.gca().invert_yaxis()
        custom_lines = [Line2D([0], [0], color=('yellow'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('orange'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('red'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('blue'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('green'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('violet'), lw=4, marker='o', linestyle=''),
                        Line2D([0], [0], color=('black'), lw=4, marker='o', linestyle='')]
                        # Line2D([0], [0], color=('grey'), lw=4, marker='o', linestyle='')]
        ax.legend(custom_lines, ["F1", "F4", "F1F4", "F5", "F1F5", "F4F5", "F1F4F5"])  # , "without_f"
        fig.savefig(fig_all_combi_F, bbox_inches='tight', dpi=dpi_value)
        plt.clf()


    # Plot postfilering results
    table_pstfilt = np.loadtxt(table_postfilt, skiprows=16, unpack=True, delimiter='\t')
    table_pstfilt = table_pstfilt.transpose()
    percents = table_pstfilt[4]  # percent point that have been filtered out by post-filtering

    if table_pstfilt.ndim ==1:
        ny = 1
    else:
        ny = len(table_pstfilt[0,:])

    ind = list(range(ny))

    print("plot 12")
    fig_postfilt = res_plots + '\\fig_postfilt.jpeg'

    fig, ax = plt.subplots(figsize=(20, 15))
    plt.bar(ind, percents)

    ind2 = np.array(ind); table_names2 = np.array(table_names)
    plt.xticks(ind2, table_names2, rotation=90, fontsize=13)
    plt.ylabel("Percent")
    plt.title("Percentage of filtered points per image pair")
    plt.gcf().subplots_adjust(bottom=0.3, right=0.9)
    fig = plt.gcf()
    fig.savefig(fig_postfilt, bbox_inches='tight', dpi=dpi_value)  # , bbox_inches='tight', dpi=dpi_value
    plt.clf()

    print("plot 13")

    def find_matching_index(list1, list2):
        inverse_index = {element: index for index, element in enumerate(list1)}
        return [(index, inverse_index[element])for index, element in enumerate(list2) if element in inverse_index]

    list_win_size = []
    for n in table_names2:
        list_win_size.append(n[-4:])
    list_win_size = list(set(list_win_size))  # get unique elements
    list_win_size.sort()

    if len(list_win_size) == 5:
        # For M1
        M1_list = [x for x in table_names2 if 'M1' in x]
        new_l1 = [y for y in M1_list if list_win_size[0] in y]  # for each window_size, it selects all filter combinations
        new_l2 = [y for y in M1_list if list_win_size[1] in y]
        new_l3 = [y for y in M1_list if list_win_size[2] in y]
        new_l4 = [y for y in M1_list if list_win_size[3] in y]
        new_l5 = [y for y in M1_list if list_win_size[4] in y]
        new_l1.sort(); new_l2.sort(); new_l3.sort(); new_l4.sort(); new_l5.sort();

        tpl1 = find_matching_index(new_l1, table_names2)
        idx1 = [x[0] for x in tpl1]

        all_names_M1 = [""] + new_l1 + [""] + new_l2 + [""] + new_l3 + [""] + new_l4 + [""] + new_l5
        idx = np.array(idx1)
        tot_bars_M1 = np.empty(0)
        i=0
        while i < 5:
            new_b = percents[idx]
            tot_bars_M1 = hstack((tot_bars_M1, 0, new_b))
            idx = idx + 1
            i=i+1

        # For M2
        M2_list = [x for x in table_names2 if 'M2' in x]
        new_l1 = [y for y in M2_list if list_win_size[0] in y]  # for each window_size, it selects all filter combinations
        new_l2 = [y for y in M2_list if list_win_size[1] in y]
        new_l3 = [y for y in M2_list if list_win_size[2] in y]
        new_l4 = [y for y in M2_list if list_win_size[3] in y]
        new_l5 = [y for y in M2_list if list_win_size[4] in y]
        new_l1.sort(); new_l2.sort(); new_l3.sort(); new_l4.sort(); new_l5.sort();

        tpl1 = find_matching_index(new_l1, table_names2)
        idx1 = [x[0] for x in tpl1]

        all_names_M2 = [""] + new_l1 + [""] + new_l2 + [""] + new_l3 + [""] + new_l4 + [""] + new_l5
        idx = np.array(idx1)
        tot_bars_M2 = np.empty(0)
        i = 0
        while i < 5:
            new_b = percents[idx]
            tot_bars_M2 = hstack((tot_bars_M2, 0, new_b))
            idx = idx + 1
            i = i + 1

        # For M3
        M3_list = [x for x in table_names2 if 'M3' in x]
        new_l1 = [y for y in M3_list if list_win_size[0] in y]  # for each window_size, it selects all filter combinations
        new_l2 = [y for y in M3_list if list_win_size[1] in y]
        new_l3 = [y for y in M3_list if list_win_size[2] in y]
        new_l4 = [y for y in M3_list if list_win_size[3] in y]
        new_l5 = [y for y in M3_list if list_win_size[4] in y]
        new_l1.sort(); new_l2.sort(); new_l3.sort(); new_l4.sort(); new_l5.sort();

        tpl1 = find_matching_index(new_l1, table_names2)
        idx1 = [x[0] for x in tpl1]

        all_names_M3 = [""] + new_l1 + [""] + new_l2 + [""] + new_l3 + [""] + new_l4 + [""] + new_l5
        idx = np.array(idx1)
        tot_bars_M3 = np.empty(0)
        i = 0
        while i < 5:
            new_b = percents[idx]
            tot_bars_M3 = hstack((tot_bars_M3, 0, new_b))
            idx = idx + 1
            i = i + 1

        # plot figure
        fig_postfilt = res_plots + '\\fig_postfilt3M1.jpeg'
        fig, ax = plt.subplots()  # M1
        plt.bar(range(0, len(tot_bars_M1)), tot_bars_M1)

        plt.xticks(range(0, len(tot_bars_M1)), all_names_M1, rotation=90, fontsize=10)
        plt.gcf().subplots_adjust(bottom=0.3, right=0.9)
        plt.ylabel("Percent")
        plt.ylim((0, 100))
        plt.title("Percentage of filtered points per image pair M1")
        plt.gcf().subplots_adjust(bottom=0.3, right=0.9)
        fig.savefig(fig_postfilt, bbox_inches='tight', dpi=dpi_value)  # , bbox_inches='tight', dpi=dpi_value
        plt.clf()
        #---
        fig_postfilt = res_plots + '\\fig_postfilt3M2.jpeg'
        fig, ax = plt.subplots()  # M2
        plt.bar(range(0, len(tot_bars_M2)), tot_bars_M2)
        plt.xticks(range(0, len(tot_bars_M2)), all_names_M2, rotation=90, fontsize=10)
        plt.ylim((0, 100))
        plt.ylabel("Percent")
        plt.title("Percentage of filtered points per image pair M2")
        plt.gcf().subplots_adjust(bottom=0.3, right=0.9)
        fig = plt.gcf()
        fig.savefig(fig_postfilt, bbox_inches='tight', dpi=dpi_value)  # , bbox_inches='tight', dpi=dpi_value
        plt.clf()

            # ---
        fig_postfilt = res_plots + '\\fig_postfilt3M3.jpeg'
        fig, ax = plt.subplots()  # M3
        plt.bar(range(0, len(tot_bars_M3)), tot_bars_M3)
        plt.xticks(range(0, len(tot_bars_M3)), all_names_M3, rotation=90, fontsize=10)
        plt.ylim((0, 100))
        plt.ylabel("Percent")
        plt.title("Percentage of filtered points per image pair M3")
        plt.gcf().subplots_adjust(bottom=0.3, right=0.9)
        fig = plt.gcf()
        fig.savefig(fig_postfilt, bbox_inches='tight', dpi=dpi_value)  # , bbox_inches='tight', dpi=dpi_value
        plt.clf()

    # end if window size == 5

    plt.close('all')

    return