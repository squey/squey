#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${JNI_INCLUDE_DIRS})

add_library(pvrush_jni SHARED PVRushJNI.cpp)
target_link_libraries(pvrush_jni ${PVKERNEL_LIBRARY} ${JAVA_JVM_LIBRARY})
