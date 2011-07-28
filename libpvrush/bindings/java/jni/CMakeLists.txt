# First, compile the Java class that binds PVRush
add_subdirectory(java)

# Then, compile the JNI shared library
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/include)
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/include)
include_directories(AFTER ${Picviz_Inspector_SOURCE_DIR}/libpvrush/src/include)
include_directories(AFTER ${JNI_INCLUDE_DIRECTORIES})

link_directories(${Picviz_Inspector_SOURCE_DIR}/libpvrush/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvfilter/src/ ${Picviz_Inspector_SOURCE_DIR}/libpvcore/src/)

add_library(pvrush_jni SHARED cpp/PVRushJNI.cpp)
target_link_libraries(pvrush_jni pvrush pvfilter pvcore ${JAVA_JVM_LIBRARY})
