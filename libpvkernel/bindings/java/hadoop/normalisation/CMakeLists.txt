project(libpvrush_bindings_java_hadoop_norm Java)

# FIXME: that should not be necessary
remove_definitions(-Wall -Wextra -pipe -ggdb3)
remove_definitions(${OpenMP_CXX_FLAGS})

# This is necessary in order to use our classes between them !!
include_directories(AFTER ../../jni/java/PVRushJNI.jar)
include_directories(AFTER CMakeFiles/HadoopNormalisation.dir)

add_library(HadoopNormalisation PVMapper.java PVJob.java)
