#!/bin/bash

# \file launch_local.sh
#
# Copyright (C) Picviz Labs 2010-2012

rm -rf /data/squid.log.out
CLASSPATH=../../jni/java/PVRushJNI.jar:/usr/share/java/commons-vfs.jar /opt/hadoop/bin/hadoop  jar HadoopNormalisation.jar org.picviz.hadoop.job.norm.PVJob -libjars ../../jni/java/PVRushJNI.jar /data/squid.log.5000.utf8 /data/squid.log.out /data/squid.format
