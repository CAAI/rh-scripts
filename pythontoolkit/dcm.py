#!/usr/bin/env python
import os
try:
    import pydicom as dicom
except ImportError:
    import dicom
import configparser
import glob
from shutil import copyfile

from rhscripts.conversion import findExtension

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
        caai_dir = os.environ['CAAI']
        config_path = '%s/share/config.ini' % caai_dir
        if not os.path.exists(config_path):
            print('You do not have a config.ini file in %s' % caai_dir)
            return
        config = configparser.ConfigParser()
        config.sections()
        config.read(config_path)
        
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