#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /opt/hadoop/hadoop-common-0.21.0.jar)
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /opt/hadoop/hadoop-common-test-0.21.0.jar)
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /opt/hadoop/hadoop-hdfs-0.21.0.jar)
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /opt/hadoop/hadoop-hdfs-0.21.0-sources.jar)
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /opt/hadoop/hadoop-hdfs-ant-0.21.0.jar)
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /opt/hadoop/hadoop-hdfs-test-0.21.0.jar)
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /opt/hadoop/hadoop-hdfs-test-0.21.0-sources.jar)
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /opt/hadoop/hadoop-mapred-0.21.0.jar)
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /opt/hadoop/hadoop-mapred-0.21.0-sources.jar)
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /opt/hadoop/hadoop-mapred-examples-0.21.0.jar)
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /opt/hadoop/hadoop-mapred-test-0.21.0.jar)
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /opt/hadoop/hadoop-mapred-tools-0.21.0.jar)
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /opt/hadoop/lib/commons-lang-2.5.jar)

add_subdirectory(normalisation)
