#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

# Compile the Java class that binds PVRush, and include our library
# This libpvrush_jni.so is a symlink to the real one. That's the only trick
# I've found to have it at to the root of our JAR archive.

# Set the libraries that need to be redistribute with our JAR archive
set(JAR_LIBS libpvrush_jni.so ${PVKERNEL_LIBRARY}.1 ${PVKERNEL_PLUGINS_LIST})

# We create an archive that will be put into the final JAR. At the initialisation
# of PVRushJNI, that archive will be extracted into a temporary directory, and then
# the libs will be used for PVKernel !
set(ARCHIVE_LIBS "pvkernel_libs.tar")
add_custom_target(libpvkernel_libs_archive "tar" "--transform" "'s,\\(.*\\)/,,'" "-hcf" "${ARCHIVE_LIBS}" ${JAR_LIBS}
				  DEPENDS ${JAR_LIBS})

# This is for debugging purpose
set(CMAKE_JAR_MAIN_CLASS "org.picviz.jni.PVRush.PVRushJNI")
set(CMAKE_JAVA_INCLUDE_PATH ${CMAKE_JAVA_INCLUDE_PATH} /usr/share/java/commons-vfs.jar /usr/share/java/jna.jar)
add_jar(PVRushJNI POSIX.java PVRushJNI.java ${ARCHIVE_LIBS})
add_dependencies(PVRushJNI libpvkernel_libs_archive)


# Launch javah on our class
add_custom_command(TARGET PVRushJNI
                   POST_BUILD
                   COMMAND javah -jni -o ../cpp/PVRushJNI.h -classpath ${PVRushJNI_CLASS_DIR} org.picviz.jni.PVRush.PVRushJNI)
