#!/bin/bash

# \file javah.sh
#
# Copyright (C) Picviz Labs 2010-2012

javac PVRushJNI.java
#javah -jni -o ../cpp/PVRushJNI.h PVRushJNI
javah -jni PVRushJNI
rm PVRushJNI.class
