#!/usr/bin/env python
import sys
import os, glob
import pydicom as dicom
from pydicom.filereader import InvalidDicomError #For rtx2mnc
from pydicom import dcmread
from nipype.interfaces.dcm2nii import Dcm2niix
import nibabel as nib
from pathlib import Path
import numpy as np
import time, warnings
from rhscripts.dcm import (
    generate_SeriesInstanceUID,
    generate_SOPInstanceUID,
    to_rtx,
    read_rtx,
    get_sort_files_dict
)
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
        Used to determine how to read the input array, options:
            'minc','nifty','torchio'

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
    # return a dict with keys={0..#slices -1}, values=Path to dcm file
    dcm_slices = get_sort_files_dict(dcmcontainer)

    # Special behavior of get_sort_files:
    # when multiple volumes are found, the top level keys will be series UIDs for the volumnes and not numbers
    if 0 not in dcm_slices:
        raise Exception(f"Multiple DICOM volumes were found in {dcmcontainer}. Please ensure that all dicom files belong to the same volume")

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
        assert len(np_array.shape) == 4, "Numpy array must be 4D for nifty conversion with 4D dicom container"
        totalSlicesInArray = np_array.shape[2]*np_array.shape[3]
    if (from_type == 'nifty' or from_type == 'torchio') and not is_4D:
        totalSlicesInArray = np_array.shape[2]

    if verbose:
        print("Checkinf if the number of files ( {} ) equals number of slices ( {} )".format(len(dcm_slices), totalSlicesInArray))
    assert len(dcm_slices) == totalSlicesInArray, "Number of files (%d) does not match number of slices (%d) in the numpy array. Please check the input." % (len(dcm_slices), totalSlicesInArray)

    ## Prepare for MODIFY HEADER
    newSIUID = generate_SeriesInstanceUID()

    # Prepare output folder
    if isinstance(dicomfolder, str):
        dicomfolder = Path(dicomfolder)
    dicomfolder.mkdir(parents=True, exist_ok=True)

    # List files, do not need to be ordered
    for SliceNumber, f in dcm_slices.items():
        ds = dcmread(f)
        # Sometimes this needs to be flipped,
        # e.g. max(dcm_slices.keys())-SliceNumber-1 (or similar).
        # This should be triggered by a dicom tag somewhere.
        i = int(SliceNumber)

        # Get single slice
        if from_type == 'minc' and is_4D:
            assert ds.pixel_array.shape == (np_array.shape[2],np_array.shape[3])
            data_slice = np_array[i // numberofslices,i % numberofslices,:,:]
        elif from_type == 'minc' and not is_4D:
            assert ds.pixel_array.shape == (np_array.shape[1],np_array.shape[2])
            data_slice = np_array[i,:,:].astype('double')
        elif from_type == 'nifty' and not is_4D:
            i = int(ds.InstanceNumber)
            assert ds.pixel_array.shape == (np_array.shape[0],np_array.shape[1])
            data_slice = np.flip(np_array[:, :, -1 * i].T, 0).astype('double')
        elif from_type == 'torchio' and not is_4D:
            data_slice = np_array[:, :, i].astype('double')
        elif from_type == 'nifty' and is_4D:
            assert ds.pixel_array.shape == (np_array.shape[0],np_array.shape[1]), "Shape of the nifty array ({}) does not match the shape of the dicom slice {}".format(np_array.shape, ds.pixel_array.shape)
            data_slice = np_array[:, :, i % numberofslices, i // numberofslices].astype('double')
        else:
            sys.exit('You must specify a from_type when using to_dcm function')

        # Check for Data Rescale
        if hasattr(ds, 'RescaleSlope'):
            # Calculate new rescale slope if needed
            if forceRescaleSlope or (np.max(data_slice) - ds.RescaleIntercept )/ds.RescaleSlope > _CONSTANTS[data_type]:
                ds.RescaleSlope = ( np.max(np_array)-ds.RescaleIntercept + 0.1 ) / float(_CONSTANTS[data_type])
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
        if 'LargestImagePixelValue' in ds:
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
               clamp_upper: int=None,
               flip: bool=False):
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
    flip : boolean, optional
        If set, the x-axis will be flipped

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

    if flip:
        np_minc = np.flip(np_minc, axis=0)

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

def mnc_to_rtx( mncfile: str,
                dcmcontainer: str,
                out_folder: str,
                out_filename: str,
                roi_names: list=None,
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
            out_filename=out_filename,roi_names=roi_names,verbose=verbose)

def nii_to_rtx( niifile: str,
                dcmcontainer: str,
                out_folder: str,
                out_filename: str,
                roi_names: list=None,
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
            out_filename=out_filename,roi_names=roi_names,verbose=verbose)

def rtx_to_mnc(dcmfile,
               mnc_container_file,
               mnc_output_file,
               behavior: str='default',
               verbose=False,
               copy_name=False):

    """Convert dcm file (RT struct) to minc file

    Parameters
    ----------
    dcmfile : string
        Path to the dicom file (RT struct)
    mnc_container_file : string
        Path to the minc file that is the container of the RT struct
    mnc_output_file : string
        Path to the minc output file
    behavior : string
        Choose how to convert to polygon. Options: default, mirada
    verbose : boolean, optional
        Default = False (if true, print info)
    copy_name : boolean, optional
        Default = False, If true the ROI name from Mirada is stored in Minc header
    Examples
    --------
    >>> from rhscripts.conversion import rtx_to_mnc
    >>> rtx_to_mnc('RTstruct.dcm','PET.mnc','RTstruct.mnc',verbose=False,copy_name=True)
    """
    import pyminc.volumes.factory as pyminc
    volume = pyminc.volumeFromFile(mnc_container_file)

    # Read RTX file. Returns dict of dict with outer key=ROI_index and inner_keys "ROIname" and "data"
    ROI_output = read_rtx( dcmfile=dcmfile,
                           img_size=volume.data.shape,
                           fn_world_to_voxel=volume.convertWorldToVoxel,
                           behavior=behavior,
                           verbose=verbose )

    for ROI_id,ROI in ROI_output.items():
        RTMINC_outname = mnc_output_file if len(ROI_output) == 1 else mnc_output_file[:-4] + "_" + str(ROI_id) + ".mnc"
        RTMINC = pyminc.volumeLikeFile(mnc_container_file,RTMINC_outname)
        RTMINC.data = ROI['data']
        RTMINC.writeFile()
        RTMINC.closeVolume()

        if copy_name:
            print('minc_modify_header -sinsert dicom_0x0008:el_0x103e="'+ROI['ROIname']+'" '+RTMINC_outname)
            os.system('minc_modify_header -sinsert dicom_0x0008:el_0x103e="'+ROI['ROIname']+'" '+RTMINC_outname)
    volume.closeVolume()

def rtx_to_nii(dcmfile,
               nii_container_file,
               nii_output_file,
               behavior: str='default',
               verbose=False,
               copy_name=False):

    """Convert dcm file (RT struct) to nifty file

    Parameters
    ----------
    dcmfile : string
        Path to the dicom file (RT struct)
    nii_container_file : string
        Path to the nifty file that is the container of the RT struct
    nii_output_file : string
        Path to the nifty output file
    behavior : string
        Choose how to convert to polygon. Options: default, mirada
    verbose : boolean, optional
        Default = False (if true, print info)
    copy_name : boolean, optional
        Default = False, If true the ROI name from Mirada is stored in Nifty header
    Examples
    --------
    >>> from rhscripts.conversion import rtx_to_nii
    >>> rtx_to_nii('RTstruct.dcm','PET.nii.gz','RTstruct.nii.gz',verbose=False,copy_name=True)
    """

    # Check file ending of output assuming one of .nii and .nii.gz
    suffix_length = 4 if nii_output_file.endswith('.nii') else 7

    volume = nib.load(nii_container_file)

    # Flip to axial-first orientation (assuming axial last)
    data = volume.get_fdata()
    data = np.swapaxes(data, 0, 2)

    # Read RTX file. Returns dict of dict with outer key=ROI_index and inner_keys "ROIname" and "data"
    ROI_output = read_rtx( dcmfile=dcmfile,
                           img_size=data.shape,
                           fn_world_to_voxel=lambda x: nib.affines.apply_affine(aff=np.linalg.inv(volume.affine),pts=x),
                           behavior=behavior,
                           voxel_dims=[2,0,1],
                           verbose=verbose )

    for ROI_id,ROI in ROI_output.items():
        # Swap back to axial last
        ROI_data = np.swapaxes(ROI['data'], 2, 0)
        RTNII_outname = nii_output_file if len(ROI_output) == 1 else nii_output_file[:-suffix_length] + "_" + str(ROI_id) + ".nii.gz"
        RTNII = nib.Nifti1Image(ROI_data,volume.affine)
        nib.save(RTNII,RTNII_outname)

    # TODO: Write name of ROI to nii tag if possible.. See rtx_to_mnc.

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
