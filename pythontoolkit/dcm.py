#!/usr/bin/env python
import os
try:
    import pydicom as dicom
except ImportError:
    import dicom
import configparser
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