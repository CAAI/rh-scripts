
cmake_minimum_required(VERSION 2.8)


INSTALL(PROGRAMS

  __init__.py
  version.py  
  conversion.py
  utils.py
  dcm.py
  lmparser.py

  DESTINATION rhscripts )

#INSTALL(PROGRAMS lmparser.py
#  PERMISSIONS
#    OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP EXECUTE WORLD_READ WORLD_EXECUTE
#  DESTINATION rhscripts)

# Legacy from when script had a conflicting name
if(EXISTS "${CMAKE_INSTALL_PREFIX}/rhscripts/dicom.py")
	message("Removing legacy file: rhscripts/dicom.py")
	file(REMOVE "${CMAKE_INSTALL_PREFIX}/rhscripts/dicom.py")
endif()
