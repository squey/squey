# Compile the Java class that binds PVRush, and include our library
# This libpvrush_jni.so is a symlink to the real one. That's the only trick
# I've found to have it at to the root of our JAR archive.

# This is for debugging purpose
set(CMAKE_JAR_MAIN_CLASS "org.picviz.jni.PVRush.PVRushJNI")
add_jar(PVRushJNI PVRushJNI.java libpvrush_jni.so)


# Launch javah on our class
add_custom_command(TARGET PVRushJNI
                   POST_BUILD
                   COMMAND javah -jni -o ../cpp/PVRushJNI.h -classpath ${PVRushJNI_CLASS_DIR} org.picviz.jni.PVRush.PVRushJNI)
