# This is necessary in order to use our classes between them !!
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} ../../jni/java/PVRushJNI.jar)

add_jar(HadoopNormalisation PVMapper.java PVJob.java)
