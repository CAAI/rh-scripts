project (rh-scripts)

cmake_minimum_required (VERSION 2.8)

set( CMAKE_INSTALL_PREFIX "/opt/caai" )

# Add python-lib
ADD_SUBDIRECTORY( pythontoolkit )

# Install programs into bin
INSTALL(PROGRAMS conversion/dicom_to_minc.py
	conversion/rtx2mnc.py
	conversion/mnc2dcm.py
	conversion/rtdose2mnc.py
  PERMISSIONS
    OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ  WORLD_EXECUTE WORLD_READ
  DESTINATION
    bin )
    
# Move config file, if it exists, to share folder
INSTALL(FILES dicom/config-default.ini DESTINATION share)

# Install PATH and PYTHONPATH configuration file
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/toolkit-config.sh.cmake ${CMAKE_CURRENT_BINARY_DIR}/toolkit-config.sh @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/toolkit-config.sh DESTINATION .)