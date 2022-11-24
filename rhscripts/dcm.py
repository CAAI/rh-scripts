#!/usr/bin/env python
import os, math
import pydicom as dicom
import configparser
import glob
from shutil import copyfile
import datetime
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, Union
from pydicom import dcmread
from pydicom.filereader import InvalidDicomError #For rtx2mnc
import cv2
import random
import socket
from rhscripts.version import __version__
import nibabel as nib


def __generate_uid_suffix() -> str:
    """ Generate and return a new UID """
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

def generate_StudyInstanceUID() -> str:
    """ Generate and return a new StudyInstanceUID """
    return '1.3.51.0.1.1.10.143.20.159.{}.7754590'.format(__generate_uid_suffix())

def generate_SeriesInstanceUID() -> str:
    """ Generate and return a new SeriesInstanceUID """
    return '1.3.12.2.1107.5.2.38.51014.{}11111.0.0.0'.format(__generate_uid_suffix())

def generate_SOPInstanceUID(i: int) -> str:
    """ Generate and return a new SOPInstanceUID

    Parameters
    ----------
    i : int
        Running number, typically InstanceNumber of the slice
    """
    return '1.3.12.2.1107.5.2.38.51014.{}{}'.format(__generate_uid_suffix(),i)

class Anonymize:
    """
    Anonymize script for DICOM file or folder containing dicom files
    Simply removes or replaces patient sensitive information.

    from rhscripts.dcm import Anonymize
    anon = Anonymize()
    anon.anonymize_folder(dicom_original_folder,dicom_anonymized_folder)
    """

    def __init__( self, verbose: bool=False, remove_private_tags: bool=False, sort_by_instance_number: bool=False ):
        """
        Parameters
        ----------
        verbose : bool, optional
            Print progress. The default is False.
        remove_private_tags : bool, optional
            Remove the private tags. The default is False.
        sort_by_instance_number : bool, optional
            Overwrites the output file to contain InstanceNumber ("dicom<04d:InstanceNumber>.dcm"). The default is False.
        """
        self.verbose = verbose
        self.remove_private_tags = remove_private_tags
        self.sort_by_instance_number = sort_by_instance_number

    def anonymize_dataset(self, dataset: dicom.dataset.Dataset, new_person_name: str="anonymous",
                  studyInstanceUID: str=None, seriesInstanceUID: str=None, replaceUIDs: bool=False) -> dicom.dataset.Dataset:
        """ Anonymize a single slice

        Parameters
        ----------
        filename: str
            Dicom file to be read and anonymized
        output_filename: str, optional
            Dicom file to be written when file is saved
        new_person_name: str, optional
            Name to replace all PN tags as well as PatientID
        studyInstanceUID: str, optional
            Overwrite instead of generating new. Used when processing multiple series one by one.
        seriesInstanceUID: str, optional
            Overwrite instead of generating new. Used to make sure all slices have same SeriesInstanceUID.
        replaceUIDs: bool, optional
            Forces replacement of UIDs. Must be True when studyInstanceUID or seriesInstanceUID is set.

        Return
        ------
        Anonymized dicom dataset
        """

        new_person_name = "anonymous" if new_person_name == None else new_person_name

        def __PN_callback(ds, data_element):
            """Called from the dataset "walk" recursive function for all data elements."""
            if data_element.VR == "PN":
                data_element.value = new_person_name

        # Remove patient name and any other person names
        dataset.walk(__PN_callback)

        # Remove data elements (should only do so if DICOM type 3 optional)
        for name in ['OtherPatientIDs', 'OtherPatientIDsSequence']:
            if name in dataset:
                delattr(dataset, name)

        # Same as above but for blanking data elements that are type 2.
        for name in ['PatientBirthDate','PatientAddress','PatientTelephoneNumbers']:
            if name in dataset:
                dataset.data_element(name).value = ''

        # Overwrite PatientID
        dataset.data_element('PatientID').value = new_person_name

        # Overwrite AccessionNumber and StudyID
        if 'AccessionNumber' in dataset:
            dataset.data_element('AccessionNumber').value = new_person_name
        if 'StudyID' in dataset:
            dataset.data_element('StudyID').value = new_person_name

        # Remove private tags
        if self.remove_private_tags:
            dataset.remove_private_tags()

        # Replace InstanceUIDs
        if replaceUIDs:
            dataset.StudyInstanceUID = studyInstanceUID if studyInstanceUID is not None else generate_StudyInstanceUID()
            dataset.SeriesInstanceUID = seriesInstanceUID if seriesInstanceUID is not None else generate_SeriesInstanceUID()
            dataset.SOPInstanceUID = generate_SOPInstanceUID(dataset.InstanceNumber)

        return dataset

    def anonymize_file( self, filename: str, output_filename: str, new_person_name: str="anonymous",
                  studyInstanceUID: str=None, seriesInstanceUID: str=None, replaceUIDs: bool=False):
        """ Anonymize a single slice

        Parameters
        ----------
        filename: str
            Dicom file to be read and anonymized
        output_filename: str
            Dicom file to be written
        """
        # Load the current dicom file to 'anonymize'
        ds = dicom.read_file(filename)

        if replaceUIDs:
            assert studyInstanceUID is not None # Must be set on folder level
            assert seriesInstanceUID is not None # Must be set on folder level

        # Anonymize
        ds = self.anonymize_dataset( dataset=ds, new_person_name=new_person_name, studyInstanceUID=studyInstanceUID,
                                     seriesInstanceUID=seriesInstanceUID, replaceUIDs=replaceUIDs )

        # Overwrite filename
        if self.sort_by_instance_number:
            output_filename = str(Path(output_filename).parent.joinpath("dicom"+str(ds.InstanceNumber).zfill(4)+'.dcm'))

        # write the 'anonymized' DICOM out under the new filename
        ds.save_as(output_filename)

    def anonymize_folder(self,foldername: str,output_foldername: str,
                         new_person_name: str="anonymous", overwrite_ending: bool=False,
                         ending_suffix: str='.dcm', studyInstanceUID: str=None,
                         replaceUIDs: bool=False) -> str:
        """ Function to anonymize all files in the folder and subfolders.

        Parameters
        ----------
        foldername : str
            Input folder with dcm files in root or subfolder
        output_foldername : str
            Output folder. Will be created if it doesnt exist.
        new_person_name : str, optional
            Name to replace all PN tags as well as PatientID
        overwrite_ending : bool, optional
            Replace the extension to .dcm or whatever the input file uses. The default is False.
        ending_suffix : str, optional
            Set the suffix manually. The default is '.dcm'.
        studyInstanceUID : str, optional
            Overwrite instead of generating new. If set, remember to set replaceUIDs. The default is None.
        replaceUIDs : bool, optional
            Forces replacement of UIDs. Must be True when studyInstanceUID is set. The default is False.

        Raises
        ------
        IOError
            When output is not a directory.

        """

        if os.path.exists(output_foldername):
            if not os.path.isdir(output_foldername):
                raise IOError("Input is directory; output name exists but is not a directory")
        else: # out_dir does not exist; create it.
            os.makedirs(output_foldername)

        if len(os.listdir(foldername)) > 9999:
            exit('Too many files in folder for script..')

        # Check for replaceUIDs:
        if replaceUIDs:
            if studyInstanceUID is None:
                studyInstanceUID = generate_StudyInstanceUID()
            seriesInstanceUID = generate_SeriesInstanceUID()
        else:
            seriesInstanceUID = None  # Default when not overwriting

        for fid,filename in enumerate(os.listdir(foldername)):
            ending = '.dcm' if os.path.splitext(filename)[1]=='' else os.path.splitext(filename)[1]
            if overwrite_ending:
                ending = ending_suffix
            filename_out = "dicom"+str(fid+1).zfill(4)+ending
            if not os.path.isdir(os.path.join(foldername, filename)):
                if self.verbose:
                    print(filename + " -> " + filename_out + "...")
                self.anonymize_file(os.path.join(foldername, filename), os.path.join(output_foldername, filename_out),new_person_name,studyInstanceUID=studyInstanceUID,seriesInstanceUID=seriesInstanceUID,replaceUIDs=replaceUIDs)
                if self.verbose:
                    print("done\r")
            else:
                if self.verbose:
                    print("Found",filename,"\r")
                self.anonymize_folder(os.path.join(foldername, filename),os.path.join(output_foldername, filename),new_person_name,studyInstanceUID=studyInstanceUID,replaceUIDs=replaceUIDs)

        return studyInstanceUID

""" ANONYMIZE END """

def get_suv_constants(
    file:Union[str, Path, dicom.Dataset],
    overwrite_values:Dict=None
) -> Tuple[Dict, Callable[[float], float]]:
    """ Extract the constants used for SUV normalization
    Parameters
    ----------
    file: 
        Path to string with dicom dataset, or pre-loaded dicom dataset
    
    Returns
    -------
    d: dict
        Dict with relevant information to perform SUV normalization
    fn: Callable
        Function to perform SUV normalization given a value
    
    """
    if isinstance(file, (str, Path)):
        ds = dcmread(file)
    else:
        ds = file

    # Injection time
    inj_time = ds[0x54,0x16][0][0x18,0x1072].value

    # Scan time
    acq_time = ds.AcquisitionTime

    # Injected dose
    dose = int(ds[0x54,0x16][0][0x18,0x1074].value)
    if dose > 999999:
        dose /= 1000000.0 # Convert to MBq

    # Halflife pr minute
    halflife = float(ds[0x54,0x16][0].RadionuclideHalfLife) / 60

    # Time diff
    d1 = datetime.datetime.strptime(inj_time, '%H%M%S.%f')
    d2 = datetime.datetime.strptime(acq_time, '%H%M%S.%f')
    diff = (d2 - d1).total_seconds() / 60

    # Reduced dose
    reduced_dose = dose*math.exp(math.log(2)/halflife*-diff)
        
    # Weight
    weight = int(ds[0x10, 0x1030].value)

    if overwrite_values is not None:
        # Manually overwrite values - might be relevant for dose, weight and post injection time
        for k, v in overwrite_values.items():
            if k == "weight":
                weight = v
            elif k == "dose":
                dose = v
            elif k == "diff":
                diff = v
    # SUV conversion
    fn = lambda x: (x * weight) / (reduced_dose * 1000)

    d = {
        'inj_time': inj_time,
        'acq_time': acq_time,
        'post_injection_time': diff,
        'injected_dose': dose,
        'halflife': halflife,
        'corrected_dose': reduced_dose,
        'weight': weight
    }
    
    return d, fn


def get_description(file):
    """Get the SeriesDescription of a dicom file

    Parameters
    ----------
    file : string
        Path to the dicom file
    """
    return dicom.read_file(file).SeriesDescription

def get_seriesnumber(file):
    """Get the SeriesNumber of a dicom file

    Parameters
    ----------
    file : string
        Path to the dicom file
    """
    return dicom.read_file(file).SeriesNumber

def get_patientid(file):
    """Get the PatientID of a dicom file

    Parameters
    ----------
    file : string
        Path to the dicom file
    """
    return dicom.read_file(file).PatientID

def get_patientname(file):
    """Get the PatientName of a dicom file

    Parameters
    ----------
    file : string
        Path to the dicom file
    """
    return dicom.read_file(file).PatientName

def get_studydate(file):
    """Get the StudyDate of a dicom file

    Parameters
    ----------
    file : string
        Path to the dicom file
    """
    return dicom.read_file(file).StudyDate

def get_time_slices(file):
    """ Get the NumberOfTimeSlices of a dicom file

    Parameters
    ----------
    file : string
        Path to the dicom file
    """
    return int(dicom.read_file(file).NumberOfTimeSlices)

def get_tag(file,tag):
    """ Get a tag from a dicom file

    Parameters
    ----------
    file : string
        Path to the dicom file
    tag : string
        Tag name
    """
    return dicom.read_file(file, force=True).data_element(tag).value


def get_reference_seriesUID_from_RTSS(file: Union[str, Path, dicom.dataset.Dataset]) -> str:
    """ Get the SeriesInstanceUID of the dataset that the RTSS is associated to """
    if not isinstance(file, dicom.dataset.Dataset):
        if not os.path.exists(str(file)):
            return None
        file = dcmread(file)
    return file.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID


def get_AC_series_info_from_PET(file: Union[str, Path, dicom.dataset.Dataset]) -> Tuple[str, str, str]:
    """ Get the StudyInstanceUID and SeriesInstanceUID from a PET dataset
    See https://marketing.webassets.siemens-healthineers.com/e43a3d1665dd2046/3141915b25bb/10751414-EKL-001-01-VG80-Osprey.pdf
    
    Parameters
    ----------
    file: str, Path, or dicom dataset
        Path to a file containing a dicom dataset, or a pre-loaded dicom series

    Returns
    -----------
    CT_used_for: str
        The purposed of the referenced CT - can be ACCT or Matching anatomy (and likely also ACCT).
        If dataset is not found (when filepath is supplied) the returned code is "No such file".
        If the correct tags used to extract the information is not present in the dataset, "Missing tag" will be returned.
    StudyInstanceUID: str
        The study ID of the ACCT series used for PET attenuation correction
    SeriesInstanceUID: str
        The Series ID of the ACCT series used for PET attenuation correction
    """
    if not isinstance(file, dicom.dataset.Dataset):
        if not os.path.exists(str(file)):
            return 'No such file', '', ''
        file = dcmread(file)

    priv_tag = (0x08, 0x1250)
    if not priv_tag in file:
        return 'Missing tag', '', ''
    StudyInstanceUID = file[priv_tag][0][0x20,0xd].value
    SeriesInstanceUID = file[priv_tag][0][0x20,0xe].value
    if (CT_code := file[priv_tag][0][(0x40,0xa170)][0][0x8,0x100].value) == '122401':
        CT_used_for = 'SameAnatomy'
    elif CT_code == '122403':
        CT_used_for = 'ACCT'
    else:
        CT_used_for = 'Unknown'
    return CT_used_for, StudyInstanceUID, SeriesInstanceUID


def sort_files(path):
    """ Sort a folder of DICOM files
    It will rename the files based on InstanceNumber
    It will create subfolders if multiple time-points exists

    Parameters
    ----------
    path : string
        Path to the dicom files
    """
    folder = '%s_sorted' % path
    if not os.path.exists(folder):
        os.mkdir(folder)

    last_file = glob.glob(path+'/*')[-1]

    do_split = False if get_time_slices(last_file) == 1 else True

    for dcmfile in os.listdir(path):

        if do_split:
            frame_name = 'frame_%010d' % int(get_tag(os.path.join(path,dcmfile),'FrameReferenceTime'))
            if not os.path.exists(os.path.join(folder,frame_name)):
                os.mkdir(os.path.join(folder,frame_name))

            fname = '%s/%s/dicom_%04d.dcm' % (folder, frame_name, get_tag(os.path.join(path,dcmfile),'InstanceNumber'))
        else:
            fname = '%s/dicom_%04d.dcm' % (folder, get_tag(os.path.join(path,dcmfile),'InstanceNumber'))

        copyfile(os.path.join(path,dcmfile), fname)


def _get_fastest_dim(lst):
    """
    lst is [ [x1,y1,z1], [x2,y2,z2], ... ]
    Will return the index of the dimension with the largest difference
    between max and min
    """
    return np.argmax(list(map(lambda x: max(x)-min(x),
                              map(list, zip(*[value for value in lst])))))


def get_sort_files_dict(path, reduce_if_only_one=True):
    """ Run through all files in a directory and return a dict of files sorted
        by their ImagePositionPatient coordinate. Multiple scans will have
        multiple keys in dict.

    Parameters
    ----------
    path : string, Path
        Path to the dicom files

    Returns
        dict{ SeriesInstanceUID: dict{ ind: path_to_file } }
    """
    path_dict = {}
    position_dict = {}

    if isinstance(path, str):
        path = Path(path)

    for p in path.rglob('*'):
        if p.name.startswith('.'):
            continue
        if not p.is_file():
            continue
        try:
            ds = dcmread(str(p))
            image_position = '_'.join([str(IPP) for IPP in
                                       ds.ImagePositionPatient])
            if ds.SeriesInstanceUID not in path_dict:
                path_dict[ds.SeriesInstanceUID] = {}
                position_dict[ds.SeriesInstanceUID] = []

            # Add file path and position to dicts
            path_dict[ds.SeriesInstanceUID][image_position] = p
            position_dict[ds.SeriesInstanceUID].append(ds.ImagePositionPatient)
        except Exception as e:
            print(f"Skipping {p}. Not dicom?. Got error: {e}")
            pass

    # Keys are x_y_z. Find fastest varying and set key to that, e.g. y only
    presorted_dict = {}
    for data_key, d in path_dict.items():
        presorted_dict[data_key] = {}
        slice_dimension = _get_fastest_dim(position_dict[data_key])
        for key, v in d.items():
            new_key = float(key.split('_')[slice_dimension])
            presorted_dict[data_key][new_key] = v
    # Sort the files by ImagePositionPatient
    sorted_dict = {}
    for data_key, d in presorted_dict.items():
        sorted_dict[data_key] = {}
        for ind, key in enumerate(sorted(d)):
            sorted_dict[data_key][ind] = d[key]

    if reduce_if_only_one and len(sorted_dict) == 1:
        # Return only the inner dict, since there is only one series present
        return next(iter(sorted_dict.values()))

    return sorted_dict


def send_data(folder, server=None, checkForEndings=True):
    """Send a dicom dataset to a dicom node

    Parameters
    ----------
    folder : string
        Path to the dicom files
    server : string, optional
        Name of the server to send to
    checkForEndings : boolean, optional
        Check if folder contains any files with dicom endings

    """
    from rhscripts.conversion import findExtension
    f = findExtension(folder)
    if not checkForEndings or isinstance(f,str):

        # Setup
        if os.environ.get('CAAI') is not None:
            caai_dir = os.environ['CAAI']
            config_path = '%s/share/config.ini' % caai_dir
            if not os.path.exists(config_path):
                print('You do not have a config.ini file in %s' % caai_dir)
                return
            config = configparser.ConfigParser()
            config.sections()
            config.read(config_path)
        else:
            print('CAAI install path is not set in your environment')
            return

        if not server:
            print('You need to select a server from config.ini')
            return

        server = server.lower()

        if not server in config:
            print('config.ini does not contain %s' % server)
            return


        cmd = 'storescu --scan-directories -aet %s -aec %s %s %s %s' % (
                config['DEFAULT']['AET'],
                config[server]['AEC'],
                config[server]['addr'],
                config[server]['port'],
                folder)

        os.system(cmd)

def replace_container(in_folder: str, container: str, out_folder: str, SeriesNumber: int=None, SeriesDescription: str=None):
    """

    Parameters
    ----------
    in_folder : str
        Path to folder with files containing PixelData that should be kept.
    container : str
        Path to folder with files that should be used as container, getting only PixelData replaced
        (alongside UIDs and optionally SeriesNumber and SeriesDescription).
    out_folder : str
        Folder to store resulting output.
    SeriesNumber : int, optional
        Overwrite the number of the series. The default is None.
    SeriesDescription : str, optional
        Overwrite the name of the series. The default is None.

    """

    def sort_files(p):
        return {dcmread(str(d)).InstanceNumber : d for d in Path(p).iterdir() if not d.name.startswith('.') }

    # Get dictionary with key=InstanceNumber val=Path-object for the file
    d_new = sort_files(in_folder)
    d_container = sort_files(container)

    # Create output folder
    Path(out_folder).mkdir(exist_ok=True,parents=True)

    seriesInstanceUID = generate_SeriesInstanceUID()

    # For each slice
    # Check start of instance numbers
    assert min(d_new.keys()) == min(d_container.keys())
    assert max(d_new.keys()) == max(d_container.keys())
    assert len(d_new) == len(d_container)
    for i in d_new.keys():

        ds_container=dcmread(str(d_container[i]))
        ds_new=dcmread(str(d_new[i]))

        ds_container.PixelData = ds_new.PixelData

        ds_container.SeriesInstanceUID = seriesInstanceUID
        ds_container.SOPInstanceUID = generate_SOPInstanceUID( i )

        if SeriesDescription is not None: ds_container.SeriesDescription = SeriesDescription
        if SeriesNumber is not None: ds_container.SeriesNumber = SeriesNumber

        ds_container.save_as(f'{out_folder}/dicom_{i:04d}.dcm')

def to_rtx(np_roi: np.ndarray,
           dcmcontainer: str,
           out_folder: str,
           out_filename: str,
           roi_names: list=None,
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
    roi_names : list
        Set the ROI names
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
        import sys
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

    for i in range(int(np_roi.max())):

        ROI_expanded = (np_roi == i+1).astype('uint8')

        #polylines = get_polylines(ROI_expanded[:,:,:,i],M,len(dcm_list)) # Get polylines.
        polylines = get_polylines(ROI_expanded,M,len(dcm_list)) # Get polylines.

        roi_set = Dataset()
        roi_set.ROINumber = str(i+1) # (3006,0022) ROI Number
        roi_set.ReferencedFrameOfReferenceUID = dicom_header_first.FrameOfReferenceUID # (3006,0024) Referenced Frame of Reference UID
        if roi_names is not None and len(roi_names) == int(np_roi.max()):
            roi_set.ROIName = roi_names[i]
        else:
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

def read_rtx( dcmfile: str, img_size: Tuple[int,int,int],
              fn_world_to_voxel: Callable, behavior: str='default',
              voxel_dims: Tuple[int,int,int]=[0,2,1],
              verbose: bool=False ) -> Dict[int, Dict[str,np.ndarray]]:
    """Read dcm file (RT struct) to dict of np.arrays - one for each ROI

    Parameters
    ----------
    dcmfile : string
        Path to the dicom file (RT struct)
    img_size : (int,int,int)
        Size of the reference volume that the RT struct was defined on.
        Assuming (slice,width,height)
    fn_world_to_voxel : string
        Function to convert world coordinate to voxel coordinate
    behavior : string
        Chose how to convert to polygon. Options: default, mirada
    flip_read_direction : boolean
        ...
    verbose : boolean, optional
        Default = False (if true, print info)
    Examples
    --------
    >>> from rhscripts.conversion import read_rtx
    >>> read_rtx('RTstruct.dcm',[127,344,344],volume.convertWorldToVoxel,verbose=False)
    """
    try:
        RTSS = dicom.read_file(dcmfile)
        ROIs = RTSS.ROIContourSequence
        ROI_output = {}

        if verbose:
            print(RTSS.StructureSetROISequence[0].ROIName)
            print("Found",len(ROIs),"ROIs")

        for ROI_id,ROI in enumerate(ROIs):

            data = np.zeros(img_size)
            contour_sequences = ROI.ContourSequence
            contour_points = {}

            if verbose:
                print(" --> Found",len(contour_sequences),"contour sequences for ROI:",RTSS.StructureSetROISequence[ROI_id].ROIName)

            for contour in contour_sequences:
                assert contour.ContourGeometricType == "CLOSED_PLANAR"

                if verbose:
                    print("\t",contour.ContourNumber,"contains",contour.NumberOfContourPoints)

                world_coordinate_points = np.array(contour.ContourData)
                world_coordinate_points = world_coordinate_points.reshape((contour.NumberOfContourPoints,3))
                voxel_coordinates_inplane = np.zeros((len(world_coordinate_points),2))
                current_slice_i = 0
                for wi,world in enumerate(world_coordinate_points):
                    voxel = fn_world_to_voxel([-world[0],-world[1],world[2]])
                    current_slice_i = voxel[abs(voxel_dims[0])]
                    # If voxel dim is negative, flip the direction
                    if math.copysign(1,voxel_dims[0]) == -1:
                        current_slice_i = img_size[0]-current_slice_i-1
                    voxel_coordinates_inplane[wi,:] = [voxel[voxel_dims[1]],voxel[voxel_dims[2]]]

                    # Track the contour points as well in float
                    if wi == 0 and not int(round(current_slice_i)) in contour_points:
                        contour_points[int(round(current_slice_i))] = []
                    contour_points[int(round(current_slice_i))].append([voxel[voxel_dims[1]],voxel[voxel_dims[2]]])
                current_slice_inner = np.zeros((img_size[1],img_size[2]),dtype=np.float32)
                if behavior == 'default':
                    # Locate each voxel covered by, or located inside, a polygon contour
                    converted_voxel_coordinates_inplane = np.array(np.round(voxel_coordinates_inplane),np.int32)
                    cv2.fillPoly(current_slice_inner,pts=[converted_voxel_coordinates_inplane],color=1)
                elif behavior == 'mirada':
                    # Check for each voxel if its center is inside polygon
                    for x in range( img_size[1] ):
                        for y in range( img_size[2] ):
                            current_slice_inner[ x, y ] = cv2.pointPolygonTest(
                                    np.array( [voxel_coordinates_inplane], dtype='float32' ),
                                    (y,x), # Not sure if should be swapped, or needs a 0.5 voxel offset for center?
                                    False
                            )
                    current_slice_inner = (current_slice_inner>0).astype('uint8')
                data[int(round(current_slice_i))] += current_slice_inner

            # Remove even areas - implies a hole.
            data[data % 2 == 0] = 0

            ROI_output[ROI_id] = {'ROIname': RTSS.StructureSetROISequence[ROI_id].ROIName,
                                  'data': data,
                                  'contour_points': contour_points}
        return ROI_output
    except InvalidDicomError:
        print("Could not read DICOM RTX file", dcmfile)
        exit(-1)


def read_rtx_v2(dcmfile: str, img_size: Tuple[int, int, int],
                fn_world_to_voxel: Callable, affine: np.ndarray,
                behavior: str = 'default', verbose: bool = False) \
                -> Dict[int, Dict[str, np.ndarray]]:
    """Read dcm file (RT struct) to dict of np.arrays - one for each ROI

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    THIS FUNCTION IS EXPERIMENTAL AND UNDER DEVELOPMENT TO 
    POTENTIALLY REPLACE THE DEFAULT read_rtss FUNCTION ABOVE
    USE WITH CAUSION - AND COMPARE TO EXPECTED OUTCOME
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Parameters
    ----------
    dcmfile : string
        Path to the dicom file (RT struct)
    img_size : (int,int,int)
        Size of the reference volume that the RT struct was defined on.
        Assuming (slice,width,height)
    fn_world_to_voxel : string
        Function to convert world coordinate to voxel coordinate
    behavior : string
        Chose how to convert to polygon. Options: default, mirada
    flip_read_direction : boolean
        ...
    verbose : boolean, optional
        Default = False (if true, print info)
    Examples
    --------
    >>> from rhscripts.conversion import read_rtx_v2
    >>> read_rtx_v2(
            'RTstruct.dcm',
            [127,344,344],
            volume.convertWorldToVoxel, # When using minc
            affine=container.affine)
    """
    try:
        RTSS = dicom.read_file(dcmfile)
        ROIs = RTSS.ROIContourSequence
        ROI_output = {}

        # # Determine reading directions
        # voxel_dims = [None, None, None]
        # codes = nib.aff2axcodes(affine)
        # # Get the slice dimension
        # if codes[2] in ('R', 'L'):  # Sagittal
        #     pass
        # elif codes[2] in ('S', 'I'):  # Axial
        #     pass
        # elif codes[2] in ('P', 'A'):  # Coronal
        #     pass
        # else:
        #     raise ValueError('Wrong affine file..')

        if verbose:
            print(RTSS.StructureSetROISequence[0].ROIName)
            print("Found", len(ROIs), "ROIs")

        for ROI_id, ROI in enumerate(ROIs):

            data = np.zeros(img_size)
            contour_sequences = ROI.ContourSequence
            contour_points = {}

            if verbose:
                print(" --> Found", len(contour_sequences), "contour sequences for ROI:", RTSS.StructureSetROISequence[ROI_id].ROIName)

            for contour in contour_sequences:
                assert contour.ContourGeometricType == "CLOSED_PLANAR"

                if verbose:
                    print("\t",contour.ContourNumber,"contains",contour.NumberOfContourPoints)

                world_coordinate_points = np.array(contour.ContourData)
                world_coordinate_points = world_coordinate_points.reshape((contour.NumberOfContourPoints,3))
                voxel_coordinates_inplane = np.zeros((len(world_coordinate_points),2))
                current_slice_i = 0
                for wi, world in enumerate(world_coordinate_points):
                    voxel = fn_world_to_voxel([-world[0], -world[1], world[2]])
                    current_slice_i = voxel[2]
                    voxel_coordinates_inplane[wi, :] = [voxel[1], voxel[0]]

                    # Track the contour points as well in float
                    k = int(round(current_slice_i))
                    if k not in contour_points:
                        contour_points[k] = []
                    contour_points[k].append([voxel[1],voxel[0]])
                current_slice_inner = np.zeros((img_size[0],img_size[1]),dtype=np.float32)
                if behavior == 'default':
                    # Locate each voxel covered by, or located inside, a polygon contour
                    converted_voxel_coordinates_inplane = np.array(np.round(voxel_coordinates_inplane),np.int32)
                    cv2.fillPoly(current_slice_inner,pts=[converted_voxel_coordinates_inplane],color=1)
                elif behavior == 'mirada':
                    # Check for each voxel if its center is inside polygon
                    for x in range( img_size[1] ):
                        for y in range( img_size[0] ):
                            current_slice_inner[ x, y ] = cv2.pointPolygonTest(
                                    np.array([voxel_coordinates_inplane], dtype='float32'),
                                    (y,x), # Not sure if should be swapped, or needs a 0.5 voxel offset for center?
                                    False
                            )
                    current_slice_inner = (current_slice_inner>0).astype('uint8')
                data[:,:,int(round(current_slice_i))] += current_slice_inner

            # Remove even areas - implies a hole.
            data[data % 2 == 0] = 0

            ROI_output[ROI_id] = {'ROIname': RTSS.StructureSetROISequence[ROI_id].ROIName,
                                  'data': data,
                                  'contour_points': contour_points}
        return ROI_output
    except InvalidDicomError:
        print("Could not read DICOM RTX file", dcmfile)
        exit(-1)

def read_rtx_v3( dcmfile: str, img_size: Tuple[int,int,int],
              fn_world_to_voxel: Callable, behavior: str='default',
              voxel_dims: Tuple[int,int,int]=[0,2,1],
              verbose: bool=False ) -> Dict[int, Dict[str,np.ndarray]]:
    """Read dcm file (RT struct) to dict of np.arrays - one for each ROI

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    THIS FUNCTION IS EXPERIMENTAL AND UNDER DEVELOPMENT TO 
    POTENTIALLY REPLACE THE DEFAULT read_rtss FUNCTION ABOVE
    USE WITH CAUSION - AND COMPARE TO EXPECTED OUTCOME
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Parameters
    ----------
    dcmfile : string
        Path to the dicom file (RT struct)
    img_size : (int,int,int)
        Size of the reference volume that the RT struct was defined on.
        Assuming (width,height,slice)
    fn_world_to_voxel : string
        Function to convert world coordinate to voxel coordinate
    behavior : string
        Chose how to convert to polygon. Options: default, mirada
    flip_read_direction : boolean
        ...
    verbose : boolean, optional
        Default = False (if true, print info)
    Examples
    --------
    >>> from rhscripts.conversion import read_rtx
    >>> read_rtx('RTstruct.dcm',[127,344,344],volume.convertWorldToVoxel,verbose=False)
    """
    try:
        RTSS = dicom.read_file(dcmfile)
        ROIs = RTSS.ROIContourSequence
        ROI_output = {}

        if verbose:
            print(RTSS.StructureSetROISequence[0].ROIName)
            print("Found",len(ROIs),"ROIs")

        for ROI_id,ROI in enumerate(ROIs):

            data = np.zeros(img_size)
            contour_sequences = ROI.ContourSequence
            contour_points = {}

            if verbose:
                print(" --> Found",len(contour_sequences),"contour sequences for ROI:",RTSS.StructureSetROISequence[ROI_id].ROIName)

            for contour in contour_sequences:
                assert contour.ContourGeometricType == "CLOSED_PLANAR"

                if verbose:
                    print("\t",contour.ContourNumber,"contains",contour.NumberOfContourPoints)

                world_coordinate_points = np.array(contour.ContourData)
                world_coordinate_points = world_coordinate_points.reshape((contour.NumberOfContourPoints,3))
                voxel_coordinates_inplane = np.zeros((len(world_coordinate_points),2))
                current_slice_i = 0
                for wi, world in enumerate(world_coordinate_points):
                    voxel = fn_world_to_voxel([-world[0],-world[1],world[2]])
                    current_slice_i = voxel[2]
                    voxel_coordinates_inplane[wi,:] = [voxel[0],voxel[1]]

                    # Track the contour points as well in float
                    if wi == 0 and not int(round(current_slice_i)) in contour_points:
                        contour_points[int(round(current_slice_i))] = []
                    contour_points[int(round(current_slice_i))].append([voxel[0],voxel[1]])
                current_slice_inner = np.zeros((img_size[0],img_size[1]),dtype=np.float32)
                if behavior == 'default':
                    # Locate each voxel covered by, or located inside, a polygon contour
                    converted_voxel_coordinates_inplane = np.array(np.round(voxel_coordinates_inplane),np.int32)
                    cv2.fillPoly(current_slice_inner,pts=[converted_voxel_coordinates_inplane],color=1)
                elif behavior == 'mirada':
                    # Check for each voxel if its center is inside polygon
                    for x in range( img_size[0] ):
                        for y in range( img_size[1] ):
                            current_slice_inner[ x, y ] = cv2.pointPolygonTest(
                                    np.array( [voxel_coordinates_inplane], dtype='float32' ),
                                    (x,y), # Not sure if needs a 0.5 voxel offset for center?
                                    False
                            )
                    current_slice_inner = (current_slice_inner>0).astype('uint8')
                data[:, :, int(round(current_slice_i))] += current_slice_inner

            # Remove even areas - implies a hole.
            data[data % 2 == 0] = 0

            ROI_output[ROI_id] = {'ROIname': RTSS.StructureSetROISequence[ROI_id].ROIName,
                                  'data': data,
                                  'contour_points': contour_points}
        return ROI_output
    except InvalidDicomError:
        print("Could not read DICOM RTX file", dcmfile)
        exit(-1)
