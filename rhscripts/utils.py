#!/usr/bin/env python

import os
import itertools
import numpy as np
import sys, random, typing, time, pydicom
from pathlib import Path
import pandas as pd
from math import sqrt

def listdir_nohidden(path):
    """List dir without hidden files
    
    Parameters
    ----------
    path : string
        Path to folder with files
    """
    return [f for f in os.listdir(path) if not f.startswith('.')]

def bbox_ND(img):
    """Get bounding box for a mask with N dimensionality
    
    Parameters
    ----------
    img : numpy array or python matrix
    """
    N = img.ndim
    out = []
    for ax in itertools.combinations(range(N), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)

# Boolean parse check for input args
def ParseBoolean(b):
    b = b.lower()
    if b == 'true':
        return True
    elif b == 'false':
        return False
    else:
        raise ValueError('Cannot parse string into boolean.')
        
class LMParser:
    """ LMParser 
    
    # Logic is used by lmparser.py
    
    ### You can also load in this function to other script, e.g. this simple case:
        from rhscripts.utils import LMParser
        parser=LMParser('llm.ptd')
        parser.save_dicom('llm.dcm')
        parser.close()
    
    """
    def __init__( self, ptd_file: str, out_folder: str=None, anonymize: bool=False, verbose: bool=False):
        self.start_time = time.time()
        # Constants
        self.LMDataID = "LARGE_PET_LM_RAWDATA"
        self.LMDataIDLen = len(self.LMDataID)
        self.LONG32BIT = 4
        self.BUFFERSIZE = 0x100 # 1 kb (c++ uses 1 mb, but this is much slower in python..)
        # Input args
        self.filename = Path( ptd_file )
        self.out_folder = Path(out_folder if out_folder is not None else self.filename.parent)
        self.out_folder.mkdir(parents=True,exist_ok=True)
        self.anonymize = anonymize
        self.verbose = verbose
        # Globals
        self.BytesReady = 0
        self.LMBuffer = None
        self.BytesToRead = None
        self.PROMPT, self.DELAY, self.KEEP, self.TOSS, self.EVENT_WORD, self.TAG_WORD = 0,0,0,0,0,0 # Counters
        self.do_chop=False
        # Setup and open PTD file
        self.__open_ptd_file()
        self.__read_dicom_header()        
        
        
    def __update_header( self ):
        import tempfile
        temp_filename = next(tempfile._get_candidate_names())

        ds = pydicom.filereader.dcmread(pydicom.filebase.DicomBytesIO(self.DicomBuffer))
        tag = ds[0x29, 0x1010]
        partial = tag.value
        start_string = 0
        end_string = 0
        for ind, l in enumerate(partial.decode().split('\n')):
            start_string = end_string
            end_string+=len(l)+1
            if l.startswith('tracer activity'):
                injected_dose_e = l.split(':=')[1].split('\r')[0]
                injected_dose = float(injected_dose_e)
                retained_injected_dose = injected_dose * float( self.retain / 100.0 )
                break
        self.__print('{:.1f} MBq -> {:.1f} MBq at {} retain value (in %)'.format(injected_dose/1000000, retained_injected_dose/1000000, self.retain))

        retained_injected_dose_e = '{:.3e}'.format(retained_injected_dose)

        if not len(injected_dose_e) == len(retained_injected_dose_e):
            self.__print("Modified DicomHeaderLength\n\tfrom {}".format(self.DicomHeaderLength))
            self.DicomHeaderLength += len(retained_injected_dose_e)-len(injected_dose_e)
            self.__print("\tTo: {}".format(self.DicomHeaderLength))

        new_string = l.replace(injected_dose_e, retained_injected_dose_e)+'\n'
        ds[0x29, 0x1010].value = partial.replace(partial[start_string:end_string], str.encode(new_string))
        ds.save_as(temp_filename)
        with open(temp_filename, "rb") as raw:
            self.DicomBuffer = raw.read(self.DicomHeaderLength)
        Path(temp_filename).unlink()  

    def chop( self, retain: int=None, out_filename: str=None, seed: int=11, rb82: bool=False):
        # Input args
        self.do_chop = True
        self.out_filename = out_filename
        self.retain = retain
        # Scaling parameter for promts/randoms
        self.rb82 = rb82
        self.seed = seed
        # Open OutFile for writing
        self.OutFile = open( self.__generate_output_name() ,'wb')
        # Scale retain to 0-1.
        retain_fraction = float( self.retain / 100.0 )
        # Setup random
        random.seed(self.seed)
        # Setup LM file
        self.__prepare_lm_file()
        # Check if lm file is LISTMODE
        assert self.get_type_from_dicom() == 'PET_LISTMODE_T'
        
        for word in self.__read_list():
            int_word = int.from_bytes(word,byteorder='little')
            if (int_word & 0x80000000) == 0x80000000:
                self.TAG_WORD += 1
                if (int_word >> 28 & 0xe) == 0x8:
                    self.OutFile.write(word)
                    if (listms := int_word & 0x1fffffff) > 0 and listms % 10000 == 0:
                        self.__print(f"Finished {listms/1000} seconds")
            else:
                self.EVENT_WORD += 1

                if int_word >> 30 == 0x1: 
                    self.PROMPT += 1 
                else: 
                    self.DELAY += 1
                """ END PROMPT/DELAY """
                
                r_ = random.random()
                # Scales randoms quadratically if tracer is Rb82
                if self.rb82:
                    if (int_word >> 30 == 0x1 and r_ < retain_fraction) or ((int_word >> 30) != 0x1 and r_ < (retain_fraction)**2):
                        self.OutFile.write(word)
                        self.KEEP += 1
                    else:
                        self.TOSS += 1
                else:
                    if r_ < retain_fraction:
                        self.OutFile.write(word)
                        self.KEEP += 1
                    else:
                        self.TOSS += 1
        self.__print("Done parsing LM words")
        self.__print(f"Prompts: {self.PROMPT}\nDelays: {self.DELAY}")
        self.__print(f"TAGS: {self.TAG_WORD}\nEVENTS: {self.EVENT_WORD}")
        self.__print(f"Keep: {self.KEEP}\nToss: {self.TOSS}\nRatio: {self.KEEP/self.EVENT_WORD*100:.2f}")
        
        # Modify DICOM header
        self.__update_header()
        # Write DICOM header etc back to file
        self.__write_header()
        
    def return_LM_statistics( self ) -> pd.DataFrame:
        # Setup LM file
        self.__prepare_lm_file()
        
        dict_prompts = {0:0}
        dict_delays = {0:0}
        timestamp = 0
        for word in self.__read_list():
            int_word = int.from_bytes(word,byteorder='little')
            if (int_word & 0x80000000) == 0x80000000:
                # TAG WORD
                if (int_word >> 28 & 0xe) == 0x8:
                    if (listms := int_word & 0x1fffffff) > 0 and listms % 1000 == 0:
                        timestamp = listms/1000
                        dict_prompts[timestamp] = 0 
                        dict_delays[timestamp] = 0 
                        self.__print(f"Finished {listms/1000} seconds")
            else:
                # EVENT WORD
                if int_word >> 30 == 0x1: 
                    dict_prompts[timestamp] += 1
                else: 
                    dict_delays[timestamp] += 1 

        df = pd.DataFrame(columns=['t','type','count'])
        for k,v in dict_prompts.items():
            df = df.append({'t': k, 'type':'prompt','numEvents':v},ignore_index=True)
            df = df.append({'t': k, 'type':'delay', 'numEvents':dict_delays[k]},ignore_index=True)
        df.t = df.t.astype('float')
        df.numEvents = df.numEvents.astype('int')
        self.__print("Done parsing LM words")
        return df
        
    def close( self ):
        self.LMFile.close()
        if self.do_chop:
            self.OutFile.close()
        self.__print("Closed files")
        self.__print("Done parsing in {:.0f} seconds".format( time.time()-self.start_time))
        
    def return_converted_dicom_header( self, anonymize_id: str=None, studyInstanceUID: str=None, 
                                       seriesInstanceUID: str=None, replaceUIDs: bool=False ) -> pydicom.dataset.FileDataset:
        """
        Return the DICOM header from the PTD file.
        """
        ds = pydicom.dcmread( pydicom.filebase.DicomBytesIO( self.DicomBuffer ) )
        if self.anonymize:
            from rhscripts.dcm import Anonymize
            anon = Anonymize()
            ds = anon.anonymize_dataset(ds, new_person_name=anonymize_id, 
                                        studyInstanceUID=studyInstanceUID, 
                                        seriesInstanceUID=seriesInstanceUID, 
                                        replaceUIDs=replaceUIDs)
        return ds 
    
    # Extract DICOM header tag (0029,1008)
    def get_type_from_dicom( self ) -> str:
        return self.return_converted_dicom_header()[0x29, 0x1008].value
    
    def save_dicom( self, out_dicom: str ):
        dcm_file = self.out_folder.joinpath( out_dicom )
        self.return_converted_dicom_header().save_as( str( dcm_file.absolute() ) )
        self.__print("Saved DICOM to: {}".format(dcm_file))
        
        
    """ PRIVATE FUNCTIONS """
    
    def __set_relative_to_end( self, distance: int ) -> int:
        self.LMFile.seek( self.TotalLMFileSize - distance, os.SEEK_SET )
        
    def __read_dicom_header( self ):            
        self.__set_relative_to_end( self.LMDataIDLen+self.LONG32BIT )
        self.DicomHeaderLength = int.from_bytes(self.LMFile.read(self.LONG32BIT),'little')
        self.__set_relative_to_end( self.DicomHeaderLength+self.LMDataIDLen+self.LONG32BIT )
        self.DicomBuffer = self.LMFile.read(self.DicomHeaderLength)
        self.__print(f"Read DICOM header of length {self.DicomHeaderLength}")
        
    def __open_ptd_file(self):
        # Open file
        if not self.filename.is_file(): sys.exit("No such PTD file",self.filename)
        self.LMFile = open(self.filename,'rb')
        self.__print(f"Opened LMFile {self.filename.name}")
        
        # Read total file size
        self.LMFile.seek(0,os.SEEK_END)
        self.TotalLMFileSize = self.LMFile.tell()
            
        # Check type
        self.__set_relative_to_end( self.LMDataIDLen ) 
        if not (PTDtype := self.LMFile.read(self.LMDataIDLen).decode('utf-8')) == self.LMDataID:
            sys.exit(f'String {PTDtype} does not match {self.LMDataID}')
        self.__print(f"PTD file is {PTDtype}")
        
    def __prepare_lm_file( self ):                    
        # Rewind and prepare for reading
        self.LMFile.seek(0)
        assert self.LMFile.tell() == 0
        
        # Get listmode-part size of file
        self.BytesRemaining = self.TotalLMFileSize - self.DicomHeaderLength - self.LMDataIDLen - self.LONG32BIT 
        self.__print(f"Got LLM file size: {self.BytesRemaining}")
        
    def __write_header( self ):
        # Append DICOM header file
        self.OutFile.write( self.DicomBuffer ) # Could edit total dose in header here..
        # Write length of DICOM header
        self.OutFile.write( self.DicomHeaderLength.to_bytes(self.LONG32BIT, byteorder='little'))
        # Write LMDataID string
        self.OutFile.write( bytes(self.LMDataID,'ascii') )
        self.__print("Wrote header back to LLM file")
        
    def __generate_output_name( self ) -> str:
        return self.out_folder.joinpath(self.out_filename).absolute() if self.out_filename else \
               self.out_folder.joinpath('{}-{:.3f}.ptd'.format(self.filename.stem,self.retain)).absolute()
        
    def __read_list( self ) -> typing.Generator[bytes, None, None]:
        while self.BytesRemaining > 0 or self.BytesReady > 0:
            if self.BytesReady == 0:
                BytesToRead = self.BUFFERSIZE if self.BytesRemaining >= self.BUFFERSIZE else self.BytesRemaining
                self.LMBuffer = self.LMFile.read(BytesToRead)
                self.BytesReady = BytesToRead
                self.BytesRemaining -= BytesToRead
            word = self.LMBuffer[:self.LONG32BIT]
            self.LMBuffer = self.LMBuffer[self.LONG32BIT:]
            self.BytesReady -= self.LONG32BIT
            yield word
        
    def __print( self, message : str):
        if self.verbose: 
            print( message )
    
