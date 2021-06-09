#!/usr/bin/env python
import sys
import os, glob
try:
    import pydicom as dicom
    from pydicom.filereader import InvalidDicomError #For rtx2mnc
except ImportError:
    import dicom
    from dicom.filereader import InvalidDicomError #For rtx2mnc
from pydicom import dcmread
from nipype.interfaces.dcm2nii import Dcm2niix
import nibabel as nib
from pathlib import Path
import numpy as np
import time, warnings
import cv2
import random
import socket
from rhscripts.dcm import generate_SeriesInstanceUID, generate_SOPInstanceUID
from rhscripts.version import __version__
import datetime

def findExtension(sourcedir,extensions = [".ima", ".IMA", ".dcm", ".DCM"]):
    """Return the number of files with one of the extensions,
    or -1 no files were found, or if more than one type of extension is found

    Parameters
    ----------
    sourcedir : string
        Path to the directory to look for files with extensions
    extensions : string list, optional
        Extensions to look for, each mutually exclusive

    Notes
    -----
    If none of the folders in sourcedir contains the extensions, it will fail.

    Examples
    --------
    >>> from rhscripts.conversion import findExtension
    >>> if findExtension('folderA') != -1:
    >>>     print("Found files in folderA")
    Found files in folderA
    """
    counts = [0]*len(extensions)
    c = 0
    for ext in extensions:
        files = glob.glob(os.path.join(sourcedir,'*' + ext) )
        counts[c] = len(files)
        c += 1
    if sum(counts) > max(counts) or sum(counts) == 0:
        return -1
    else:
        return extensions[counts.index(max(counts))]

def look_for_dcm_files(folder):
    """Return first folder found with one of the extensions,
    or -1 no files were found, or if more than one type of extension is found

    Parameters
    ----------
    folder : string
        Path to the directory to crawl for files with extensions

    Notes
    -----
    Only the path to the first occurence of files will be returned

    Examples
    --------
    >>> from rhscripts.conversion import look_for_dcm_files
    >>> dicomfolder = look_for_dcm_files('folderA')
    """
    if findExtension(folder) != -1:
        return folder
    for root,subdirs,files in os.walk(folder):
        if len(subdirs) > 0:
            continue
        if not len(files) > 0:
            continue
        if findExtension(root) != -1:
            return root
    return -1

def dcm_to_mnc(folder,target='.',fname=None,dname=None,verbose=False,checkForFileEndings=True):
    """Convert a folder with dicom files to minc

    Parameters
    ----------
    folder : string
        Path to the directory to crawl for files
    target : string, optional
        Path to the install prefix
    fname : string, optional
        Name of the minc file, if not set, use minc-toolkit default
    dname : string, optional
        Name of the folder to place the minc file into, if not set, use minc-toolkit default
    verbose : boolean, optional
        Set the verbosity
    checkForFileEndings : boolean, optional
        If set, crawl for a folder with dicom file endings, otherwise just use input

    Notes
    -----


    Examples
    --------
    >>> from rhscripts.conversion import dcm_to_mnc
    >>> dcm_to_mnc('folderA',target='folderB',fname='PETCT',dname='mnc',checkForFileEndings=False)
    """
    dcmcontainer = look_for_dcm_files(folder) if checkForFileEndings else folder

    if dcmcontainer == -1:
        print("Could not find dicom files in container..")
        exit(-1)

    cmd = 'dcm2mnc -usecoordinates -clobber '+dcmcontainer+'/* '+target
    if not fname is None:
        cmd += ' -fname "'+fname+'"'
    if not dname is None:
        cmd += ' -dname '+dname

    if verbose:
        print("Command %s" % cmd)

    os.system(cmd)


def dcm_to_nifty(source_dir, output_dir, out_filename, compress='y'):
    """Converts a folder containing dicom files (slices) into one nifty file.

    Args:
        source_dir (str or Path): input directory containing .dcm or .ima files
        output_dir (str or Path): output directory where nitfy file is saved
        out_filename (str): name of the output nifty file
        compress ('y', 'n'): whether output file is .nii.gz ('y') or .nii ('n'). Default is 'y'.
    """
    # uncomment that when we have dealt with str vs path objects
    # dcmcontainer = look_for_dcm_files(source_dir)

    converter = Dcm2niix()
    converter.inputs.source_dir = source_dir
    converter.inputs.compress = compress
    converter.inputs.output_dir = output_dir
    converter.inputs.out_filename = out_filename
    try:
        converter.run()
    except Exception as e:
        print(e)


def to_dcm(np_array,
           dicomcontainer,
           dicomfolder,
           verbose=False,
           modify=False,
           description=None,
           study_id=None,
           patient_id=None,
           checkForFileEndings=True,
           forceRescaleSlope=False,
           from_type='minc'):
    """Convert a numpy array (initially loaded from minc or nifty file) to dicom

    Parameters
    ----------
    np_array : NumpyArray
        Array loaded from a file (nii or mnc)
    dicomcontainer : string
        Path to the directory containing the dicom container
    dicomfolder : string
        Path to the output dicom folder
    verbose : boolean, optional
        Set the verbosity
    modify : boolean, optional
        Create new SeriesInstanceUID and SOPInstanceUID
        Default on if description or id is set
    description : string, optional
        Sets the SeriesDescription tag in the dicom files
    id : int, optional
        Sets the SeriesNumber tag in the dicom files
    forceRescaleSlope : boolean, optional
        Forces recalculation of rescale slope
    from_type : str, optional
        Used to determine how to read the input array, options: 'minc','nifty'

    Examples
    --------
    >>> from rhscripts.conversion import to_dcm
    >>> to_dcm(array,'PETCT','PETCT_new',description="PETCT_new",id="600")
    """

    # CONSTANT(S)
    _CONSTANTS = {'int16': 32767,
                  'uint16': 65535}

    if verbose:
        print("Converting to DICOM")

    if description or study_id:
        modify = True

    if checkForFileEndings:
        dcmcontainer = look_for_dcm_files(dicomcontainer)
        if dcmcontainer == -1:
            sys.exit("Could not find dicom files in container..")
    else:
        dcmcontainer = dicomcontainer

    # change path str to path object
    if isinstance(dcmcontainer, str):
        dcmcontainer = Path(dcmcontainer)

    # gather the dicom slices from the container
    dcm_slices = [f for f in dcmcontainer.iterdir() if not f.name.startswith('.')]

    # Get information about the dataset from a single file
    ds = dcmread(dcm_slices[0])
    data_type = ds.pixel_array.dtype.name

    # Determine if this is a 4D array
    if is_4D := ( hasattr(ds, 'NumberOfSlices') and len(np_array.shape) == 4 ):
        numberofslices = ds.NumberOfSlices # Get the number of slices per time point
        if verbose:
            print("Converting a 4D array")

    # Check that the correct number of files exists
    if from_type == 'minc' and is_4D:
        totalSlicesInArray = np_array.shape[0]*np_array.shape[1]
    if from_type == 'minc' and not is_4D:
        totalSlicesInArray = np_array.shape[0]
    if from_type == 'nifty' and is_4D:
        sys.exit('Nifty 4D conversion not yet implemented')
    if from_type == 'nifty' and not is_4D:
        totalSlicesInArray = np_array.shape[2]

    if verbose:
        print("Checkinf if the number of files ( {} ) equals number of slices ( {} )".format(len(dcm_slices), totalSlicesInArray))
    assert len(dcm_slices) == totalSlicesInArray

    ## Prepare for MODIFY HEADER
    newSIUID = generate_SeriesInstanceUID()

    # Prepare output folder
    if isinstance(dicomfolder, str):
        dicomfolder = Path(dicomfolder)
    dicomfolder.mkdir(parents=True, exist_ok=True)

    # List files, do not need to be ordered
    for f in dcm_slices:
        ds = dcmread(f)
        i = int(ds.InstanceNumber)-1

        # Get single slice
        if from_type == 'minc' and is_4D:
            assert ds.pixel_array.shape == (np_array.shape[2],np_array.shape[3])
            data_slice = np_array[i // numberofslices,i % numberofslices,:,:]
        elif from_type == 'minc' and not is_4D:
            assert ds.pixel_array.shape == (np_array.shape[1],np_array.shape[2])
            data_slice = np_array[i,:,:].astype('double')
        elif from_type == 'nifty' and not is_4D:
            assert ds.pixel_array.shape == (np_array.shape[0],np_array.shape[1])
            data_slice = np.flip(np_array[:, :, -(i+1)].T, 0).astype('double')
        elif from_type == 'nifty' and is_4D:
            sys.exit('Nifty 4D conversion not yet implemented')
        else:
            sys.exit('You must specify a from_type when using to_dcm function')

        # Check for Data Rescale
        if hasattr(ds, 'RescaleSlope'):
            # Calculate new rescale slope if needed
            if forceRescaleSlope or (np.max(data_slice) - ds.RescaleIntercept )/ds.RescaleSlope > _CONSTANTS[data_type]:
                ds.RescaleSlope = ( np.max(np_array)-ds.RescaleIntercept + 0.000000000001 ) / float(_CONSTANTS[data_type])
                if verbose:
                    print(f"Setting RescaleSlope to {ds.RescaleSlope}")

            # Normalize using RescaleSlope and RescaleIntercept
            data_slice = (data_slice - ds.RescaleIntercept) / ds.RescaleSlope

        # Assert ranges
        #assert np.max(data_slice) <= _CONSTANTS[data_type], f"Data must be below absolute max ({_CONSTANTS[data_type]}) for {data_type} dicom container. Was {np.max(data_slice)}"
        if not np.max(data_slice) <= _CONSTANTS[data_type]:
            raise ValueError( f"Data must be below absolute max ({_CONSTANTS[data_type]}) for {data_type} dicom container. Was {np.max(data_slice)} after applying RescaleIntercept and RescaleSlope." )
            return
        if data_type.startswith('u'): # Only applies to unsigned, e.g. uint16
            if not np.min(data_slice) >= 0:
                raise ValueError( f"Data must be strictly positive for {data_type} dicom container. Was {np.min(data_slice)} after applying RescaleIntercept and RescaleSlope." )
                return
            #assert np.min(data_slice) >= 0, f"Data must be strictly positive for {data_type} dicom container. Was {np.min(data_slice)}"
        data_slice = data_slice.astype(data_type) # This will fail without warning if the above assertions is not meet!

        # Insert pixel-data
        ds.PixelData = data_slice.tostring()

        # Update LargesImagetPixelValue tag pr slice
        ds.LargestImagePixelValue = int(np.ceil(data_slice.max()))

        if modify:
            if verbose:
                print("Modifying DICOM headers")

            # Set information if given
            if description:
                ds.SeriesDescription = description
            if study_id:
                ds.SeriesNumber = study_id
            if patient_id:
                ds.PatientID = patient_id
                ds.PatientName = patient_id

            # Update SOP - unique per file
            ds.SOPInstanceUID = generate_SOPInstanceUID(i+1)

            # Update MediaStorageSOPInstanceUID - unique per file
            #ds.file_meta.MediaStorageSOPInstanceUID = newMSOP # Not needed anymore?

            # Same for all files
            ds.SeriesInstanceUID = newSIUID

        fname = f"dicom_{ds.InstanceNumber:04}.dcm"
        ds.save_as(dicomfolder.joinpath(fname))

    if verbose:
        print("Output written to %s" % dicomfolder)

def mnc_to_dcm(mncfile,
               dicomcontainer,
               dicomfolder,
               verbose=False,
               modify=False,
               description=None,
               study_id=None,
               checkForFileEndings=True,
               forceRescaleSlope=False,
               zero_clamp=False,
               clamp_lower: int=None,
               clamp_upper: int=None):
    """Convert a minc file to dicom

    Parameters
    ----------
    mncfile : string
        Path to the minc file
    dicomcontainer : string
        Path to the directory containing the dicom container
    dicomfolder : string
        Path to the output dicom folder
    verbose : boolean, optional
        Set the verbosity
    modify : boolean, optional
        Create new SeriesInstanceUID and SOPInstanceUID
        Default on if description or id is set
    description : string, optional
        Sets the SeriesDescription tag in the dicom files
    id : int, optional
        Sets the SeriesNumber tag in the dicom files
    forceRescaleSlope : boolean, optional
        Forces recalculation of rescale slope
    zero_clamp : boolean, optional
        Force the non-zero element in the input to be zero
    clamp_lower : int, optional
        Force a lower bound on the input data
    clamp_upper : int, optional
        Force an upper bound on the input data

    Examples
    --------
    >>> from rhscripts.conversion import mnc_to_dcm
    >>> mnc_to_dcm('PETCT_new.mnc','PETCT','PETCT_new',description="PETCT_new",id="600")
    """

    # Load the minc file
    import pyminc.volumes.factory as pyminc
    minc = pyminc.volumeFromFile(mncfile)
    np_minc = np.array(minc.data)
    minc.closeVolume()

    # Remove non-zero elements
    if zero_clamp:
        np_minc[ np_minc < 0 ] = 0.0

    # Force values to lie within a range accepted by the dicom container
    if clamp_lower is not None:
        np_minc = np.maximum( np_minc, clamp_lower )
    if clamp_upper is not None:
        np_minc = np.minimum( np_minc, clamp_upper )

    to_dcm(np_array=np_minc,
           dicomcontainer=dicomcontainer,
           dicomfolder=dicomfolder,
           verbose=verbose,
           modify=modify,
           description=description,
           study_id=study_id,
           checkForFileEndings=checkForFileEndings,
           forceRescaleSlope=forceRescaleSlope,
           from_type='minc')


""" DEPRECATED FUNCTION. """
def mnc_to_dcm_4D(*args, **kwargs):
    warnings.warn("mnc_to_dcm_4D has been replaced by mnc_to_dcm which incorporates the full functionality.",
                  DeprecationWarning)
    time.sleep(5)
    return mnc_to_dcm(*args, **kwargs)


def nifty_to_dcm(nftfile,
                 dicomcontainer,
                 dicomfolder,
                 verbose=False,
                 modify=False,
                 description=None,
                 study_id=None,
                 patient_id=None,
                 checkForFileEndings=True,
                 forceRescaleSlope=False,
                 clamp_lower: int=None,
                 clamp_upper: int=None):
    """Convert a minc file to dicom
    Parameters
    ----------
    nftfile : string or Path object
        Path to the minc file
    dicomcontainer : string or Path object
        Path to the directory containing the dicom container
    dicomfolder : string or Path object
        Path to the output dicom folder
    verbose : boolean, optional
        Set the verbosity
    modify : boolean, optional
        Create new SeriesInstanceUID and SOPInstanceUID
        Default on if description or id is set
    description : string, optional
        Sets the SeriesDescription tag in the dicom files
    study_id : int, optional
        Sets the StudyID tag in the dicom files
    patient_id : int, optional
        Sets the PatientName and PatientID tag in the dicom files
    forceRescaleSlope : boolean, optional
        Forces recalculation of rescale slope
    clamp_lower : int, optional
        Force a lower bound on the input data
    clamp_upper : int, optional
        Force an upper bound on the input data

    Examples
    --------
    >>> from rhscripts.conversion import nifty_to_dcm
    >>> nifty_to_dcm('PETCT_new.nii.gz', 'PETCT', 'PETCT_new', description="PETCT_new", id="600")
    """

    np_nifti = nib.load(nftfile).get_fdata()

    # Force values to lie within a range accepted by the dicom container
    if clamp_lower is not None:
        np_nifti = np.maximum( np_nifti, clamp_lower )
    if clamp_upper is not None:
        np_nifti = np.minimum( np_nifti, clamp_upper )

    to_dcm(np_array=np_nifti,
           dicomcontainer=dicomcontainer,
           dicomfolder=dicomfolder,
           verbose=verbose,
           modify=modify,
           description=description,
           study_id=study_id,
           patient_id=patient_id,
           checkForFileEndings=checkForFileEndings,
           forceRescaleSlope=forceRescaleSlope,
           from_type='nifty')


def rtdose_to_mnc(dcmfile,mncfile):

    """Convert dcm file (RD dose distribution) to minc file

    Parameters
    ----------
    dcmfile : string
        Path to the dicom file (RD type)
    mncfile : string
        Path to the minc file

    Examples
    --------
    >>> from rhscripts.conversion import rtdose_to_mnc
    >>> rtdose_to_mnc('RD.dcm',RD.mnc')
    """

    # Load the dicom
    ds = dicom.dcmread(dcmfile)

    # Extract the starts and steps of the x,y,z space
    starts = ds.ImagePositionPatient
    steps = [float(i) for i in ds.PixelSpacing];
    if not (ds.SliceThickness==''):
        dz = ds.SliceThickness
    elif 'GridFrameOffsetVector' in ds:
        dz = ds.GridFrameOffsetVector[1] -ds.GridFrameOffsetVector[0]
    else:
        raise IOError("Cannot determine slicethickness!")
    steps.append(dz)

    #reorder the starts and steps!
    myorder = [2,1,0]
    starts = [ starts[i] for i in myorder]
    myorder = [2,0,1]
    steps = [ steps[i] for i in myorder]
    #change the sign (e.g. starts=[1,-1,-1].*starts)
    starts = [a*b for a,b in zip([1,-1,-1],starts)]
    steps = [a*b for a,b in zip([1,-1,-1],steps)]

    #Get the pixel data and scale it correctly
    dose_array = ds.pixel_array*float(ds.DoseGridScaling)

    # Write the output minc file
    import pyminc.volumes.factory as pyminc
    out_vol = pyminc.volumeFromData(mncfile,dose_array,dimnames=("zspace", "yspace", "xspace"),starts=starts,steps=steps)
    out_vol.writeFile()
    out_vol.closeVolume()

def to_rtx(np_roi: np.ndarray,
           dcmcontainer: str,
           out_folder: str,
           out_filename: str,
           verbose: bool=False):

    """Convert label numpy array to RT struct dicom file

    Parameters
    ----------
    np_roi : np.ndarray
        Numpy array in memory to be converted. Assume integer values of [0,1(,..)]
    dcmcontainer : string
        Path to dicom container to be used
    out_folder : string
        Name of the output folder
    out_filename : string
        Name of the output dicom file
    verbose : boolean, optional
        Verbosity of function
    """

    from pydicom.sequence import Sequence
    from pydicom.dataset import Dataset, FileDataset
    from decimal import Decimal, getcontext
    getcontext().prec = 7

    # Create affine transformation matrix:
    def get_affine_transform(first_scan_path,last_scan_path,series_length):
        # Load headers.
        dicom_header_first = dicom.dcmread(first_scan_path)
        dicom_header_last = dicom.dcmread(last_scan_path)

        # Extract header info.
        IOP = dicom_header_first.ImageOrientationPatient # Orientation is the same for all slices.
        IOP = [float(i) for i in IOP]

        IPP = dicom_header_first.ImagePositionPatient # Position of first slice.
        IPP = [float(i) for i in IPP]

        IPP2 = dicom_header_last.ImagePositionPatient # Position of last slice.
        IPP2 = [float(i) for i in IPP2]

        PS = dicom_header_first.PixelSpacing # Pixel Spacing is the same for all slices.
        PS = [float(i) for i in PS]

        # Creat empty matrix M.
        M = np.zeros([4,4])

        # Fill first column. Direction and Spacing for X coordinate.
        M[0,0]=IOP[0]*PS[0]
        M[1,0]=IOP[1]*PS[0]
        M[2,0]=IOP[2]*PS[0]

        # Fill second column. Direction and Spacing for Y coordinate.
        M[0,1]=IOP[3]*PS[1]
        M[1,1]=IOP[4]*PS[1]
        M[2,1]=IOP[5]*PS[1]

        # Fill third column. Direction and Spacing for Z coordinate.
        M[0,2]=(IPP[0]-IPP2[0])/(1-series_length)
        M[1,2]=(IPP[1]-IPP2[1])/(1-series_length)
        M[2,2]=(IPP[2]-IPP2[2])/(1-series_length)

        # Fill fourth column. Coordinate shift.
        M[0,3]=IPP[0]
        M[1,3]=IPP[1]
        M[2,3]=IPP[2]
        M[3,3]=1

        return M

    # Mask --> Polyline transform.
    def get_polylines(mask,affine_transform_matrix,series_length):
        # Create empty list to fill with contours.
        polylines = []
        for i in range(series_length): # Create a list space for each input slice. There are probably better ways to do this!
            polylines.append([])

        # Extract contours from mask and save as polylines.
        for i in range(series_length):
            contours, hierarchy = cv2.findContours(mask[:,:,i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # Convert masks to contours.
            for contour in contours: # Loop through contours.
                contour_list = np.ndarray.tolist(contour) # Convert numpy array to list.
                polyline = list() # Create empty polyline container.
                for triplet in contour_list: # Loop through contour coordinate triplets.
                    triplet = triplet[0] # Modify list structure.
                    triplet.append(i+1) # Add Z dimension.
                    triplet.append(1) # Add fourth dimension for matrix multiplication.
                    triplet = list(np.matmul(affine_transform_matrix,triplet)) # Multiply coordinate with affine transform matrix.
                    triplet.pop() # Remove fourth dimension.
                    for coord in triplet: # Loop through single coordinates.
                        polyline.append(str(coord)) # Convert each number to string format for dicom header.
                polylines[i].append(polyline) # Save polyline.
                print("APPENDING")
        return polylines


    # Define filter function for incoming image series:
    def checkEqual(lst):
       return lst[1:] == lst[:-1]

    # Define function for expanding point and line delineations:
    def expand_polyline(poly_input,M):

        input_length = len(poly_input)
        x_shift = Decimal((M[0][0])/2)
        y_shift = Decimal((M[1][1])/2)

        poly_input = [Decimal(item) for item in poly_input]

        expanded_poly = list()

        # Expand point delineations:
        if input_length == 3:
            expanded_poly.append([str(poly_input[0]+x_shift),str(poly_input[1]+y_shift),str(poly_input[2])])
            expanded_poly.append([str(poly_input[0]+x_shift),str(poly_input[1]-y_shift),str(poly_input[2])])
            expanded_poly.append([str(poly_input[0]-x_shift),str(poly_input[1]+y_shift),str(poly_input[2])])
            expanded_poly.append([str(poly_input[0]-x_shift),str(poly_input[1]-y_shift),str(poly_input[2])])

            expanded_poly = [list(i) for i in set(map(tuple, expanded_poly))] # Remove duplicated triplets.

        # Expand line delineations:
        if input_length == 6:
            expanded_poly.append([str(poly_input[0]+x_shift),str(poly_input[1]+y_shift),str(poly_input[2])])
            expanded_poly.append([str(poly_input[0]+x_shift),str(poly_input[1]-y_shift),str(poly_input[2])])
            expanded_poly.append([str(poly_input[0]-x_shift),str(poly_input[1]+y_shift),str(poly_input[2])])
            expanded_poly.append([str(poly_input[0]-x_shift),str(poly_input[1]-y_shift),str(poly_input[2])])

            expanded_poly.append([str(poly_input[3]+x_shift),str(poly_input[4]+y_shift),str(poly_input[5])])
            expanded_poly.append([str(poly_input[3]+x_shift),str(poly_input[4]-y_shift),str(poly_input[5])])
            expanded_poly.append([str(poly_input[3]-x_shift),str(poly_input[4]+y_shift),str(poly_input[5])])
            expanded_poly.append([str(poly_input[3]-x_shift),str(poly_input[4]-y_shift),str(poly_input[5])])

            expanded_poly = [list(i) for i in set(map(tuple, expanded_poly))] # Remove duplicated triplets.

        polyline = list()
        for item in expanded_poly:
            polyline+=item

        return polyline

    # change path str to path object
    if isinstance(dcmcontainer, str):
        dcmcontainer = Path(dcmcontainer)

    # gather the dicom slices from the container
    dcm_list = [str(f) for f in dcmcontainer.iterdir() if not f.name.startswith('.') and f.is_file()]
    ref_dict = dict() # Store dcm image SOP Instance UID's and file name for later use.
    series_list = list() # For checking that all slices are from the same series.

    # Fill out numpy matrix and reference dictionary.
    for files in dcm_list:
        dcm_slice = dcmread(files)
        ref_dict[dcm_slice.InstanceNumber]=[dcm_slice.SOPInstanceUID,files]
        series_list.append(dcm_slice.SeriesInstanceUID)

    # Check that all dicom images in the folder belong to the same series.
    if checkEqual(series_list) == False:
        print("Error: All image slices must belong to the same series.")
        sys.exit()

    # Get first and last image from dicom series.
    first_scan_path = ref_dict[1][1]
    last_scan_path = ref_dict[len(dcm_list)][1]

    # Read dicom header from first file.
    dicom_header_first = dicom.dcmread(first_scan_path)

    # Get current time.
    time = datetime.datetime.now()

    # Get image modality to determine SOP class UID for later.
    image_modality = dicom_header_first.Modality
    if image_modality == "CT":
        SOP_class_UID = '1.2.840.10008.5.1.4.1.1.2'
    elif image_modality == "MR":
        SOP_class_UID = '1.2.840.10008.5.1.4.1.1.4'
    elif image_modality == "PT":
        #SOP_class_UID = '1.2.840.10008.5.1.4.1.1.20'
        SOP_class_UID = '1.2.840.10008.5.1.4.1.1.128'

    # Create new series instance UID.
    newSIUID = generate_SeriesInstanceUID()

    #%% Create RT file:

    # Create dicom meta header.
    file_meta = Dataset()

    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3' # (0002,0002) Media Storage SOP Class UID
    file_meta.MediaStorageSOPInstanceUID = newSIUID # (0002,0003) Media Storage SOP Instance UID
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2' # (0002,0010) Transfer Syntax UID
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.691.0.21" # (0002,0012) Implementation Class UID
    file_meta.ImplementationVersionName = '1.0' # (0002,0013) Implementation Version Name

    # Create dicom dataset.
    RTSTRUCT = FileDataset('RT',{},preamble=b"\0" * 128,file_meta=file_meta)

    # Fill dicom header using the input scan series.
    RTSTRUCT.InstanceCreationDate = time.strftime('%Y%m%d') # (0008,0012) Instance Creation Date (YYYYMMDD)
    RTSTRUCT.InstanceCreationTime = time.strftime('%H%M%S') # (0008,0013) Instance Creation Time (HHMMSS)
    RTSTRUCT.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3' # (0008,0016) SOP Class UID
    RTSTRUCT.SOPInstanceUID = newSIUID # (0008,0018) SOP Instance UID

    # Patient Information.
    RTSTRUCT.PatientName = dicom_header_first.PatientName # (0010,0010) Patient Name
    RTSTRUCT.PatientID = dicom_header_first.PatientID # (0010,0020) Patient ID
    RTSTRUCT.PatientBirthDate = dicom_header_first.PatientBirthDate # (0010,0030) Patient Birth Date
    RTSTRUCT.PatientSex = dicom_header_first.PatientSex # (0010,0040) Patient Sex

    # Study Information.
    RTSTRUCT.StudyInstanceUID = dicom_header_first.StudyInstanceUID # (0020,000d) Study Instance UID
    RTSTRUCT.StudyDate = dicom_header_first.StudyDate # (0008,0020) Study Date (YYYYMMDD)
    RTSTRUCT.StudyTime = dicom_header_first.StudyTime # (0008,0030) Study Time (HHMMSS)
    RTSTRUCT.OperatorsName = '' # (0008,1070) Operators Name
    RTSTRUCT.ReferringPhysicianName = dicom_header_first.ReferringPhysicianName # (0008,0090) Referring Physician Name
    RTSTRUCT.StudyID = dicom_header_first.StudyID # (0020,0010) Study ID
    RTSTRUCT.AccessionNumber = dicom_header_first.AccessionNumber # (0008,0050) Accession Number
    RTSTRUCT.StudyDescription = dicom_header_first.StudyDescription # (0008,1030) Study Description
    if 'ReferencedStudySequence' in dicom_header_first:
        RTSTRUCT.ReferencedStudySequence = dicom_header_first.ReferencedStudySequence # (0008,1110) Referenced Study Sequence

    # Series Information.
    RTSTRUCT.Modality = 'RTSTRUCT' # (0008,0060) Modality
    RTSTRUCT.SeriesInstanceUID = newSIUID # (0020,000e) Series Instance UID
    RTSTRUCT.SeriesNumber = str(200 + random.randint(1,200)) # (0020,0011) Series Number
    RTSTRUCT.SeriesDate = time.strftime('%Y%m%d') # (0008,0021) Series Date
    RTSTRUCT.SeriesTime = time.strftime('%H%M%S') # (0008,0031) Series Time
    RTSTRUCT.SeriesDescription = out_filename # (0008,103e) Series Description

    # Equipment Information.
    RTSTRUCT.Manufacturer = 'RH-SCRIPTS v{}'.format(__version__) # (0008,0070) Manufacturer
    RTSTRUCT.StationName = socket.gethostname() # (0008,1010) Station Name

    # Structure Set.
    RTSTRUCT.StructureSetLabel = 'ROI' # (3006,0002) Structure Set Label
    RTSTRUCT.StructureSetName = '' # (3006,0004) Structure Set Name
    RTSTRUCT.StructureSetDate = time.strftime('%Y%m%d') # (3006,0008) Structure Set Date (YYYYMMDD)
    RTSTRUCT.StructureSetTime = time.strftime('%H%M%S') # (3006,0009) Structure Set Time (HHMMSS)

    RTSTRUCT.ReferencedFrameOfReferenceSequence = Sequence([Dataset()]) # (3006,0010) Referenced Frame Of Reference Sequence
    RTSTRUCT.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID = dicom_header_first.FrameOfReferenceUID # (0020,0052) Frame Of Reference UID

    RTSTRUCT.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence = Sequence([Dataset()]) # (3006,0012) RT Referenced Study Sequence
    RTSTRUCT.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2' # (0008,1150) Referenced SOP Class UID
    if 'ReferencedStudySequence' in dicom_header_first:
        RTSTRUCT.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].ReferencedSOPInstanceUID = dicom_header_first.ReferencedStudySequence[0].ReferencedSOPInstanceUID # (0008, 1155) Referenced SOP Instance UID

    RTSTRUCT.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence = Sequence([Dataset()]) # (3006,0014) RT Referenced Series Sequence
    RTSTRUCT.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID = dicom_header_first.SeriesInstanceUID # (0020,000e) Series Instance UID

    RTSTRUCT.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence = Sequence([Dataset()]) # (3006,0016) Contour Image Sequence
    for i in range(len(ref_dict)):
        contour = Dataset()
        contour.ReferencedSOPClassUID = SOP_class_UID # (0008,1150) Referenced SOP Class UID
        contour.ReferencedSOPInstanceUID = ref_dict[i+1][0] # (0008,1155) Referenced SOP Instance UID
        if i == 0:
            RTSTRUCT.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence[0] = contour
        else:
            RTSTRUCT.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence.append(contour)

    RTSTRUCT.StructureSetROISequence = Sequence([Dataset()]) # (3006,0020)  Structure Set ROI Sequence
    RTSTRUCT.ROIContourSequence = Sequence([Dataset()]) # (3006,0039)  ROI Contour Sequence

    # Get Affine Transform Matrix.
    M=get_affine_transform(first_scan_path,last_scan_path,len(dcm_list))

    for i in range(np_roi.max()):

        ROI_expanded = (np_roi == i+1).astype(int)
        ROI_expanded = ROI_expanded.astype('uint8')

        #polylines = get_polylines(ROI_expanded[:,:,:,i],M,len(dcm_list)) # Get polylines.
        polylines = get_polylines(ROI_expanded,M,len(dcm_list)) # Get polylines.

        roi_set = Dataset()
        roi_set.ROINumber = str(i+1) # (3006,0022) ROI Number
        roi_set.ReferencedFrameOfReferenceUID = dicom_header_first.FrameOfReferenceUID # (3006,0024) Referenced Frame of Reference UID
        roi_set.ROIName = 'ROI_'+str(i+1) # (3006,0026) ROI Name
        roi_set.ROIGenerationAlgorithm = 'AUTOMATIC' # (3006,0036) ROI Generation Algorithm
        if i == 0:
            RTSTRUCT.StructureSetROISequence[0] = roi_set
        else:
            RTSTRUCT.StructureSetROISequence.append(roi_set)

        contour = Dataset()
        contour.ROIDisplayColor = ['255', '0', '0'] # (3006,002a) ROI Display Color
        contour.ReferencedROINumber = str(i+1) # (3006,0084) Referenced ROI Number
        contour.ContourSequence = Sequence([Dataset()]) # (3006,0040)  Contour Sequence
        if i == 0:
            RTSTRUCT.ROIContourSequence[0] = contour
        else:
            RTSTRUCT.ROIContourSequence.append(contour)

        contour_number = 1 # Initialize contour number.
        for x in range(len(polylines)):
            if polylines[x]:
                for n in range(len(polylines[x])):
                    poly = Dataset()
                    poly.ContourImageSequence = Sequence([Dataset()]) # (3006,0016)  Contour Image Sequence
                    poly.ContourImageSequence[0].ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2' # (0008,1150) Referenced SOP Class UID
                    poly.ContourImageSequence[0].ReferencedSOPInstanceUID = ref_dict[x+1][0] # (0008,1155) Referenced SOP Instance UID

                    if len(polylines[x][n]) == 3 or len(polylines[x][n]) == 6: # Create additional triplets.
                        polylines[x][n] = expand_polyline(polylines[x][n],M)

                    poly.ContourGeometricType = 'CLOSED_PLANAR' # (3006,0042) Contour Geometric Type
                    poly.NumberOfContourPoints = str(int(len(polylines[x][n])/3)) # (3006,0046) Number of Contour Points
                    poly.ContourNumber = str(contour_number) # (3006,0048) Contour Number
                    poly.ContourData = polylines[x][n] # (3006,0050) Contour Data

                    if contour_number == 1:
                        RTSTRUCT.ROIContourSequence[i].ContourSequence[0] = poly
                    else:
                        RTSTRUCT.ROIContourSequence[i].ContourSequence.append(poly)
                    contour_number += 1 # Update contour number.

    # Save RTSTRUCT file.
    out_path = Path(out_folder).joinpath(out_filename+".dcm")
    out_path.parent.mkdir(exist_ok=True,parents=True)
    dicom.filewriter.write_file(str(out_path), RTSTRUCT, write_like_original=False)

def mnc_to_rtx( mncfile: str,
                dcmcontainer: str,
                out_folder: str,
                out_filename: str,
                verbose: bool=False):

    """Convert minc label file to RT struct dicom file

    Parameters
    ----------
    mncfile : string
       Path to minc-file to be converted. Assume integer values of [0,1(,..)]
    dcmcontainer : string
       Path to dicom container to be used
    out_folder : string
       Name of the output folder
    out_filename : string
       Name of the output dicom file
    verbose : boolean, optional
       Verbosity of function
    """
    # Load the minc file
    import pyminc.volumes.factory as pyminc
    minc = pyminc.volumeFromFile(mncfile,labels=True)
    np_minc = np.array(minc.data,dtype='int8')
    minc.closeVolume()

    # Convert from axial-first to axial-last
    np_minc = np.swapaxes( np.swapaxes( np_minc, 0, 1), 1, 2 )
    to_rtx( np_roi=np_minc, dcmcontainer=dcmcontainer, out_folder=out_folder,
            out_filename=out_filename,verbose=verbose)

def nii_to_rtx( niifile: str,
                dcmcontainer: str,
                out_folder: str,
                out_filename: str,
                verbose: bool=False):

    """Convert minc label file to RT struct dicom file

    Parameters
    ----------
    niifile : string
       Path to nifty-file to be converted. Assume integer values of [0,1(,..)]
    dcmcontainer : string
       Path to dicom container to be used
    out_folder : string
       Name of the output folder
    out_filename : string
       Name of the output dicom file
    verbose : boolean, optional
       Verbosity of function
    """
    # Load the minc file
    np_nifti = nib.load(niifile).get_fdata()

    # Convert from axial-first to axial-last
    np_nifti = np.swapaxes( np.swapaxes( np_nifti, 0, 1), 1, 2 )
    # More needed? UNTESTED!!

    to_rtx( np_roi=np_nifti, dcmcontainer=dcmcontainer, out_folder=out_folder,
            out_filename=out_filename,verbose=verbose)

def rtx_to_mnc(dcmfile,
               mnc_container_file,
               mnc_output_file,
               verbose=False,
               copy_name=False,
               dry_run=False,
               roi_name=None,
               crop_area=False):

    """Convert dcm file (RT struct) to minc file

    Parameters
    ----------
    dcmfile : string
        Path to the dicom file (RT struct)
    mnc_container_file : string
        Path to the minc file that is the container of the RT struct
    mnc_output_file : string
        Path to the minc output file
    verbose : boolean, optional
        Default = False (if true, print info)
    copy_name : boolean, optional
        Default = False, If true the ROI name from Mirada is store in Minc header
    dry_run : boolean, optional
        Default = False, If true, only the roi names will be printed, no files are saved
    roi_name : string, optional
        Specify a name, only ROIs matching this description will be created
    crop_area : boolean, optional
        Instead of the full area matching the mnc_container, crop the area to match values > 0
    Examples
    --------
    >>> from rhscripts.conversion import rtx_to_mnc
    >>> rtx_to_mnc('RTstruct.dcm',PET.mnc','RTstruct.mnc',verbose=False,copy_name=True)
    """

    try:
        import pyminc.volumes.factory as pyminc
        RTSS = dicom.read_file(dcmfile)

        ROIs = RTSS.ROIContourSequence

        if verbose or dry_run:
            print(RTSS.StructureSetROISequence[0].ROIName)
            print("Found",len(ROIs),"ROIs")

        if not dry_run:
            volume = pyminc.volumeFromFile(mnc_container_file)


        for ROI_id,ROI in enumerate(ROIs):

            # Create one MNC output file per ROI
            RTMINC_outname = mnc_output_file if len(ROIs) == 1 else mnc_output_file[:-4] + "_" + str(ROI_id) + ".mnc"
            if not dry_run:
                RTMINC = pyminc.volumeLikeFile(mnc_container_file,RTMINC_outname)
            contour_sequences = ROI.ContourSequence

            if verbose or dry_run:
                print(" --> Found",len(contour_sequences),"contour sequences for ROI:",RTSS.StructureSetROISequence[ROI_id].ROIName)

            # Only save for ROI with specific name
            if not roi_name == None and not roi_name == RTSS.StructureSetROISequence[ROI_id].ROIName:
                if verbose:
                    print("Skipping ")
                continue

            if not dry_run:
                for contour in contour_sequences:
                    assert contour.ContourGeometricType == "CLOSED_PLANAR"

                    current_slice_i_print = 0

                    if verbose:
                        print("\t",contour.ContourNumber,"contains",contour.NumberOfContourPoints)

                    world_coordinate_points = np.array(contour.ContourData)
                    world_coordinate_points = world_coordinate_points.reshape((contour.NumberOfContourPoints,3))
                    current_slice = np.zeros((volume.getSizes()[1],volume.getSizes()[2]))
                    voxel_coordinates_inplane = np.zeros((len(world_coordinate_points),2))
                    current_slice_i = 0
                    for wi,world in enumerate(world_coordinate_points):
                        voxel = volume.convertWorldToVoxel([-world[0],-world[1],world[2]])
                        current_slice_i = voxel[0]
                        voxel_coordinates_inplane[wi,:] = [voxel[2],voxel[1]]
                    current_slice_inner = np.zeros((volume.getSizes()[1],volume.getSizes()[2]),dtype=np.float)
                    converted_voxel_coordinates_inplane = np.array(np.round(voxel_coordinates_inplane),np.int32)
                    cv2.fillPoly(current_slice_inner,pts=[converted_voxel_coordinates_inplane],color=1)

                    RTMINC.data[int(round(current_slice_i))] += current_slice_inner

            if not dry_run:
                # Remove even areas - implies a hole.
                RTMINC.data[RTMINC.data % 2 == 0] = 0

                # Save cropped area of label, or full volume
                if crop_area:
                    # TODO
                    print("Functionality not implemented yet")
                    exit(-1)
                else:
                    RTMINC.writeFile()
                    RTMINC.closeVolume()

                if copy_name:
                    print('minc_modify_header -sinsert dicom_0x0008:el_0x103e="'+RTSS.StructureSetROISequence[ROI_id].ROIName+'" '+RTMINC_outname)
                    os.system('minc_modify_header -sinsert dicom_0x0008:el_0x103e="'+RTSS.StructureSetROISequence[ROI_id].ROIName+'" '+RTMINC_outname)
        if not dry_run:
            volume.closeVolume()

    except InvalidDicomError:
        print("Could not read DICOM RTX file", dcmfile)
        exit(-1)



def hu2lac(infile,outfile,kvp=None,mrac=False,verbose=False):

    """Convert CT-HU to LAC @ 511 keV

    Parameters
    ----------
    infile : string
        Path to the input mnc file
    outfile : string
        Path to the outputmnc file
    kvp : int, optional
        Integer that specify the kVp on CT scan (overwrites the search for a value)
    mrac: boolean, optional
        if set, scales the LAC [cm^-1] by 10000
    verbose : boolean, optional
        Set the verbosity
    Examples
    --------
    >>> from rhscripts.conversion import hu2lac
    >>> hu2lac('CT_hu.mnc',CT_lac.mnc',kvp = 120)
    """
    if not kvp:
        kvp = os.popen('mincinfo -attvalue dicom_0x0018:el_0x0060 ' + infile + ' -error_string noKVP').read().rstrip()
        if kvp == 'noKVP':
            print('Cant find KVP in header. Are you sure this a CT image?')
            return
        else:
            kvp = int(kvp)
    print('kvp = ' + str(kvp))

    if mrac:
        fscalefactor = 10000
    else:
        fscalefactor = 1

    if kvp==100:
        cmd = 'minccalc -expression \"if(A[0]<52){ ((A[0]+1000)*0.000096)*'+str(fscalefactor)+'; } else { ((A[0]+1000)*0.0000443+0.0544)*'+str(fscalefactor)+'; }\" ' + infile + ' ' + outfile + ' -clobber'
    elif kvp == 120:
        cmd = 'minccalc -expression \"if(A[0]<47){ ((A[0]+1000)*0.000096)*'+str(fscalefactor)+'; } else { ((A[0]+1000)*0.0000510+0.0471)*'+str(fscalefactor)+'; }\" ' + infile + ' ' + outfile + ' -clobber'
    else:
        print('No conversion for this KVP!')
        return

    if verbose:
        print(cmd)

    os.system(cmd)


def lac2hu(infile,outfile,kvp=None,mrac=False,verbose=False):

    """Convert LAC @ 511 keV to  CT-HU

    Parameters
    ----------
    infile : string
        Path to the input mnc file
    outfile : string
        ath to the outputmnc file
    kvp : int, optional
        Integer that specify the kVp on CT scan (overwrites the search for a value)
    mrac: boolean, optional
        if set, accounts for the fact that LAC [cm^-1] is multiplyed by 10000
    verbose : boolean, optional
        Set the verbosity
    Examples
    --------
    >>> from rhscripts.conversion import lac2hu
    >>> lac2hu('CT_lac.mnc',CT_hu.mnc',kvp = 120)
    """
    if not kvp:
        kvp = os.popen('mincinfo -attvalue dicom_0x0018:el_0x0060 ' + infile + ' -error_string noKVP').read().rstrip()
        if kvp == 'noKVP':
            print('Cant find KVP in header. Are you sure this a CT image?')
            return
        else:
            kvp = int(kvp)
    print('kvp = ' + str(kvp))

    if mrac:
        fscalefactor = 10000
    else:
        fscalefactor = 1

    if kvp==100:
        breakpoint = ((52+1000)*0.000096)*fscalefactor
        cmd = 'minccalc -expression \"if(A[0]<'+str(breakpoint)+'){((A[0]/'+str(fscalefactor)+')/0.000096)-1000; } else { ((A[0]/'+str(fscalefactor)+')-0.0544)/0.0000443 - 1000; }\" ' + infile + ' ' + outfile + ' -clobber'
    elif kvp == 120:
        breakpoint = ((47+1000)*0.000096)*fscalefactor
        cmd = 'minccalc -expression \"if(A[0]<'+str(breakpoint)+'){((A[0]/'+str(fscalefactor)+')/0.000096)-1000; } else { ((A[0]/'+str(fscalefactor)+')-0.0471)/0.0000510 - 1000; }\" ' + infile + ' ' + outfile + ' -clobber'
    else:
        print('No conversion for this KVP!')
        return

    if verbose:
        print(cmd)

    os.system(cmd)
