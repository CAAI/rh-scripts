#!/usr/bin/env python
import os, sys, random, typing, time, argparse, pydicom
from pathlib import Path

""" LMParser 
Author: Claes Ladefoged
Date: 11-04-2021

### Example use cases from cmd-line ###
    
    Check LLM file is LISTMODE:
        python lmparser.py <ptd file>
    
    Extract DICOM header to file:
        python lmparser.py <ptd file> --out_dicom <dicom_filename>
        
    Chop LM file
        python lmparser.py <ptd file> <percent retained>
        
    Above will output dicom and chopped ptd file in same folder as input ptd.    
    You can specify output folder and/or output chopped name as optional inputs.
    
    Full example with arguments
        python lmparser.py llm.ptd --verbose --retain 25 --out_folder . \
               --out_filename lm_25p.ptd --out_dicom llm.dcm -seed 1337
    
### You can also load in this function to other script, e.g. this simple case:
    from lmparser import LMParser
    parser=LMParser('llm.ptd')
    parser.save_dicom('llm.dcm')
    parser.close()

---------------------------------------------------------------------------------------

Versions:
    11-04-2021: Added functionality similar to LMChopper and LMBreakout merged together.
    
Todo: 
    - Add logic to keep prompt and delay differently.
    - Implement anonymize (?)
"""

class LMParser:
    def __init__( self, ptd_file: str, out_folder: str=None, anonymize: bool=False, verbose: bool=False ):
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
        self.anonymize = anonymize # NOT YET IMPLEMENTED..
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
        
    def chop( self, retain: int=None, out_filename: str=None, seed: int=11 ):
        # Input args
        self.do_chop = True
        self.out_filename = out_filename
        self.retain = retain
        self.seed = seed
        # Open OutFile for writing
        self.OutFile = open( self.__generate_output_name() ,'wb')
        # Scale retain to 0-1.
        retain_fraction = float( self.retain / 100.0 )
        # Setup random
        random.seed(self.seed)
        # Setup LM file
        self.__prepare_lm_file()
        
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
                
                """ FROM HERE YOU CAN IMPLEMENT THE PROMPT / DELAY RETAIN DIFFERENCE LOGIC """
                if int_word >> 30 == 0x1: 
                    self.PROMPT += 1 
                else: 
                    self.DELAY += 1
                """ END PROMPT/DELAY """
                
                if random.random() < retain_fraction:
                    self.OutFile.write(word)
                    self.KEEP += 1
                else:
                    self.TOSS += 1
        self.__print("Done parsing LM words")
        self.__print(f"Prompts: {self.PROMPT}\nDelays: {self.DELAY}")
        self.__print(f"TAGS: {self.TAG_WORD}\nEVENTS: {self.EVENT_WORD}")
        self.__print(f"Keep: {self.KEEP}\nToss: {self.TOSS}\nRatio: {self.KEEP/self.EVENT_WORD*100:.2f}")
        
        # Write DICOM header etc back to file
        self.__write_header()
        
    def close( self ):
        self.LMFile.close()
        if self.do_chop:
            self.OutFile.close()
        self.__print("Closed files")
        self.__print("Done parsing in {:.0f} seconds".format( time.time()-self.start_time))
        
    def return_converted_dicom_header( self ) -> pydicom.dataset.FileDataset:
        return pydicom.dcmread( pydicom.filebase.DicomBytesIO( self.DicomBuffer ) ) 
    
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
        self.__print(f"PTD file is {self.LMDataID}")
        
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
    
#%% Main Program
def main():
    
    # INPUTS
    parser = argparse.ArgumentParser()
    parser.add_argument("ptd_file", help='Input PTD LLM file', type=str)
    parser.add_argument("--retain", help='Percent (float) of LMM events to retain (0-100)', type=float)
    parser.add_argument("--out_folder", help='Output folder for chopped PTD LLM file(s)', type=str)
    parser.add_argument("--out_filename", help='Output filename for chopped PTD LLM file', type=str)
    parser.add_argument("--seed", help='Seed value for random', default=11, type=int)
    parser.add_argument("--out_dicom", help='Save DICOM header to file', type=str)
    parser.add_argument('--anonymize', action='store_true')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    
    parser = LMParser( ptd_file = args.ptd_file,  out_folder = args.out_folder, 
                       anonymize = args.anonymize, verbose = args.verbose)
    if args.retain: parser.chop(retain = args.retain, out_filename = args.out_filename, seed = args.seed)
    if args.out_dicom: parser.save_dicom(args.out_dicom)
    parser.close()
    
if __name__ == "__main__":
    main()
