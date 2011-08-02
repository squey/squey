#!/bin/bash
javac PVRushJNI.java
javah -jni -o ../cpp/PVRushJNI.h PVRushJNI
