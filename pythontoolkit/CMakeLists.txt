
cmake_minimum_required(VERSION 2.8)

find_package( PythonInterp 3.6 REQUIRED )

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    from distutils import sysconfig as sc
    print(sc.get_python_lib(prefix='', plat_specific=True))"
  OUTPUT_VARIABLE PYTHON_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE)

message("${PYTHON_SITE}")

INSTALL(PROGRAMS
  
  conversion.py

  DESTINATION ${PYTHON_SITE}/rhscripts )
