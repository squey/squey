# First, compile the Java class that binds PVRush
add_subdirectory(java)

# Then, compile the JNI shared library
include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${JNI_INCLUDE_DIRS})

add_library(pvrush_jni SHARED cpp/PVRushJNI.cpp)
target_link_libraries(pvrush_jni ${PVKERNEL_LIBRARY} ${JAVA_JVM_LIBRARY})
