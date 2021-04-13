project (rh-scripts)

cmake_minimum_required (VERSION 2.8)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "/opt/caai" CACHE PATH "default install path" FORCE )
endif()

# Add python-lib
ADD_SUBDIRECTORY( pythontoolkit )

# Install programs into bin
INSTALL(PROGRAMS conversion/dicom_to_minc.py
	conversion/rtx2mnc.py
	conversion/mnc2dcm.py
	conversion/rtdose2mnc.py
	conversion/hu2lac.py
	conversion/lac2hu.py
	dicom/anonymize_dicom.py
  PERMISSIONS
    OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ  WORLD_EXECUTE WORLD_READ
  DESTINATION
    bin )

# Add symlinks
ADD_CUSTOM_TARGET(link_target ALL
	COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_INSTALL_PREFIX}/rhscripts/lmparser.py ${CMAKE_INSTALL_PREFIX}/bin/LMParser)
    
# Move config file, if it exists, to share folder
INSTALL(FILES dicom/config-default.ini DESTINATION share)

# Install PATH and PYTHONPATH configuration file
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/toolkit-config.sh.cmake ${CMAKE_CURRENT_BINARY_DIR}/toolkit-config.sh @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/toolkit-config.sh DESTINATION ${CMAKE_INSTALL_PREFIX})

# Add UNIT tests
INSTALL(DIRECTORY tests DESTINATION ${CMAKE_INSTALL_PREFIX})

