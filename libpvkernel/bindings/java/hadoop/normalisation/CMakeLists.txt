#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

# This is necessary in order to use our classes between them !!
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} ../../jni/java/PVRushJNI.jar)

add_jar(HadoopNormalisation org/picviz/mapreduce/output/NetworkOutputFormat.java org/picviz/mapreduce/output/TCPNetworkOutputFormat.java org/picviz/hadoop/job/norm/PVMapper.java org/picviz/hadoop/job/norm/PVReducer.java org/picviz/hadoop/job/norm/PVJob.java)
