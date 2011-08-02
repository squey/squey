project(libpvrush_bindings_java_jni Java)

# FIXME: that should not be necessary
remove_definitions(-Wall -Wextra -pipe -ggdb3)
remove_definitions(${OpenMP_CXX_FLAGS})
add_library(PVRushJNI PVRushJNI.java)

# FIXME: should we execute javah to create the JNI C interface automatically ?

