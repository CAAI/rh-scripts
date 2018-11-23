project (rh-scripts)

cmake_minimum_required (VERSION 2.8)

ADD_SUBDIRECTORY( pythontoolkit )

INSTALL(PROGRAMS conversion/dicom_to_minc.py
  PERMISSIONS
    OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ  WORLD_EXECUTE WORLD_READ
  DESTINATION
    bin )
