project (rh-scripts)

cmake_minimum_required (VERSION 2.8)

set( CMAKE_INSTALL_PREFIX "/opt/caai" )

ADD_SUBDIRECTORY( pythontoolkit )

INSTALL(PROGRAMS conversion/dicom_to_minc.py
	conversion/mnc2dcm.py
  PERMISSIONS
    OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ  WORLD_EXECUTE WORLD_READ
  DESTINATION
    bin )

INSTALL(FILES toolkit-config.sh DESTINATION .)
