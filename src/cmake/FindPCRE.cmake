# - Try to find the PCRE regular expression library
# Once done this will define
#
#  PCRE_FOUND - system has the PCRE library
#  PCRE_INCLUDE_DIR - the PCRE include directory
#  PCRECPP_INCLUDE_DIR - the C++ PCRE include directory
#  PCRE_LIBRARIES - The libraries needed to use PCRE
#  PCRECPP_LIBRARY - The libraries needed to use C++ PCRE

# Copyright (c) 2006, Alexander Neundorf, <neundorf@kde.org>
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.


#if (PCRE_INCLUDE_DIR AND PCRE_PCREPOSIX_LIBRARY AND PCRE_PCRE_LIBRARY AND PCRECPP_INCLUDE_DIR and PCRECPP_LIBRARY)
#  # Already in cache, be silent
#  set(PCRE_FIND_QUIETLY TRUE)
#endif (PCRE_INCLUDE_DIR AND PCRE_PCREPOSIX_LIBRARY AND PCRE_PCRE_LIBRARY AND PCRECPP_INCLUDE_DIR and PCRECPP_LIBRARY)


if (NOT WIN32)
  # use pkg-config to get the directories and then use these values
  # in the FIND_PATH() and FIND_LIBRARY() calls
  include(UsePkgConfig)

  pkgconfig(libpcre _PCREIncDir _PCRELinkDir _PCRELinkFlags _PCRECflags)
endif (NOT WIN32)

find_path(PCRE_INCLUDE_DIR pcre.h PATHS ${_PCREIncDir} PATH_SUFFIXES pcre)
find_path(PCRECPP_INCLUDE_DIR pcrecpp.h PATHS ${_PCREIncDir} PATH_SUFFIXES pcre)

find_library(PCRE_PCRE_LIBRARY NAMES pcre PATHS ${_PCRELinkDir})

find_library(PCRECPP_LIBRARY NAMES pcrecpp PATHS ${_PCRELinkDir})

find_library(PCRE_PCREPOSIX_LIBRARY NAMES pcreposix PATHS ${_PCRELinkDir})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PCRE DEFAULT_MSG PCRE_INCLUDE_DIR PCRE_PCRE_LIBRARY PCRE_PCREPOSIX_LIBRARY )

set(PCRE_LIBRARIES ${PCRE_PCRE_LIBRARY} ${PCRE_PCREPOSIX_LIBRARY} ${PCRECPP_LIBRARY})

mark_as_advanced(PCRE_INCLUDE_DIR PCRE_LIBRARIES PCRE_PCREPOSIX_LIBRARY PCRE_PCRE_LIBRARY PCRECPP_LIBRARY PCRECPP_INCLUDE_DIR)

