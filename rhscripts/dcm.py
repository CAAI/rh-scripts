#!/usr/bin/env python
import os
try:
    import pydicom as dicom
except ImportError:
    import dicom
import configparser
import glob
from shutil import copyfile
import datetime
from rhscripts.conversion import findExtension
from pathlib import Path
from typing import Optional

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
    
    def __generate_uid(self) -> str:
        """ Generate and return a new UID """
        UID = str(datetime.datetime.now())
        for symbol in ['-',' ',':','.']:
            UID = UID.replace(symbol,'')
        return UID
    
    def generate_StudyInstanceUID(self) -> str:
        """ Generate and return a new StudyInstanceUID """
        return '1.3.51.0.1.1.10.143.20.159.{}.7754590'.format(self.__generate_uid())
    
    def generate_SeriesInstanceUID(self) -> str:
        """ Generate and return a new SeriesInstanceUID """
        return '1.3.12.2.1107.5.2.38.51014.{}11111.0.0.0'.format(self.__generate_uid())
    
    def generate_SOPInstanceUID(self,i: int) -> str:
        """ Generate and return a new SOPInstanceUID
        
        Parameters
        ----------
        i : int
            Running number, typically InstanceNumber of the slice
        """
        return '1.3.12.2.1107.5.2.38.51014.{}{}'.format(self.__generate_uid(),i)
    
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
            dataset.StudyInstanceUID = studyInstanceUID if studyInstanceUID is not None else self.generate_StudyInstanceUID()
            dataset.SeriesInstanceUID = seriesInstanceUID if seriesInstanceUID is not None else self.generate_SeriesInstanceUID()
            dataset.SOPInstanceUID = self.generate_SOPInstanceUID(dataset.InstanceNumber)
    
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
        ds = anonymize_dataset( dataset=ds, new_person_name=new_person_name, studyInstanceUID=studyInstanceUID,
                                seriesInstanceUID=seriesInstanceUID, replaceUIDs=replaceUIDs )
        
        # Overwrite filename
        if self.sort_by_instance_number:
            output_filename = "dicom"+str(dataset.InstanceNumber).zfill(4)+'.dcm'
            
        # write the 'anonymized' DICOM out under the new filename
        ds.save_as(output_filename) 
        
    def anonymize_folder(self,foldername: str,output_foldername: str, 
                         new_person_name: str="anonymous", overwrite_ending: bool=False, 
                         ending_suffix: str='.dcm', studyInstanceUID: str=None,
                         replaceUIDs: bool=False):
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
            if studyInstanceUID is None: studyInstanceUID = self.generate_StudyInstanceUID()
            seriesInstanceUID = self.generate_SeriesInstanceUID()
    
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
                self.anonymize_folder(os.path.join(foldername, filename),os.path.join(output_foldername, filename),new_person_name,studyInstanceUID=studyInstanceUID,seriesInstanceUID=seriesInstanceUID,replaceUIDs=replaceUIDs)

""" ANONYMIZE END """

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
        
def generate_uid_suffix() -> str:
    """ Generate and return a new UID """
    UID = str(datetime.datetime.now())
    for symbol in ['-',' ',':','.']:
        UID = UID.replace(symbol,'')
    return UID

def generate_StudyInstanceUID() -> str:
    """ Generate and return a new StudyInstanceUID """
    return '1.3.51.0.1.1.10.143.20.159.{}.7754590'.format(generate_uid_suffix())
    
def generate_SeriesInstanceUID() -> str:
    """ Generate and return a new SeriesInstanceUID """
    return '1.3.12.2.1107.5.2.38.51014.{}11111.0.0.0'.format(generate_uid_suffix())

def generate_SOPInstanceUID(i: int) -> str:
    """ Generate and return a new SOPInstanceUID
    
    Parameters
    ----------
    i : int
        Running number, typically InstanceNumber of the slice
    """
    return '1.3.12.2.1107.5.2.38.51014.{}{}'.format(generate_uid_suffix(),i)
        
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
        return {dicom.read_file(str(d)).InstanceNumber : d for d in Path(p).iterdir() }

    # Get dictionary with key=InstanceNumber val=Path-object for the file
    d_new = sort_files(in_folder)
    d_container = sort_files(container)
    
    # Create output folder   
    Path(out_folder).mkdir(exist_ok=True,parents=True)
    
    # For each slice, 1-indexed due to InstanceNumber key
    for i in range(1,len(d_new)+1):
        
        ds_container=dicom.read_file(str(d_container[i]))
        ds_new=dicom.read_file(str(d_new[i]))
        
        ds_container.PixelData = ds_new.PixelData
        
        ds_container.SeriesInstanceUID = generate_SeriesInstanceUID()
        ds_container.SOPInstanceUID = generate_SOPInstanceUID( i )
        
        if SeriesDescription is not None: ds_container.SeriesDescription = SeriesDescription
        if SeriesNumber is not None: ds_container.SeriesNumber = SeriesNumber        

        ds_container.save_as(f'{out_folder}/dicom_{i:04d}.dcm')