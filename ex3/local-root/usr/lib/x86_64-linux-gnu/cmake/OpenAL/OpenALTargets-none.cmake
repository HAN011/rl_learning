#----------------------------------------------------------------
# Generated CMake target import file for configuration "None".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "OpenAL::OpenAL" for configuration "None"
set_property(TARGET OpenAL::OpenAL APPEND PROPERTY IMPORTED_CONFIGURATIONS NONE)
set_target_properties(OpenAL::OpenAL PROPERTIES
  IMPORTED_LOCATION_NONE "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libopenal.so.1.23.1"
  IMPORTED_SONAME_NONE "libopenal.so.1"
  )

list(APPEND _cmake_import_check_targets OpenAL::OpenAL )
list(APPEND _cmake_import_check_files_for_OpenAL::OpenAL "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libopenal.so.1.23.1" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
