####
# BCode benchmarks
SET(SRC_FILES
${COMMON_SRC_FILES}
${BZ_SRC_FILES}
bcode.cpp
)
add_executable(bcode ${SRC_FILES})
target_link_libraries(bcode ${COMMON_LIBS} ${PVKERNEL_LIBRARIES} ${PICVIZ_LIBRARIES})
####

####
# Mem benchmarks
SET(SRC_FILES
${COMMON_SRC_FILES}
mem.cpp
)
add_executable(mem ${SRC_FILES})
target_link_libraries(mem ${COMMON_LIBS} ${PVKERNEL_LIBRARIES})
####

####
# BZ reduce
SET(SRC_FILES
${COMMON_SRC_FILES}
${BZ_SRC_FILES}
${OCL_SRC_FILES}
../ocl/bccb.cpp
../ocl/bccb2.cpp
bred.cpp
)
#add_definitions(-DRED_STATS)
add_executable(red ${SRC_FILES})
target_link_libraries(red ${COMMON_LIBS} ${PVKERNEL_LIBRARIES} ${OCL_LIBRARIES})
####

####
# Count bits
SET(SRC_FILES
${COMMON_SRC_FILES}
count_bits.cpp
)
add_executable(bits ${SRC_FILES})
target_link_libraries(bits ${COMMON_LIBS} ${PVKERNEL_LIBRARIES})
####

####
# Reduction n
SET(SRC_FILES
${COMMON_SRC_FILES}
reduction_n.cpp
)
add_executable(nred ${SRC_FILES})
target_link_libraries(nred ${COMMON_LIBS} ${PVKERNEL_LIBRARIES})
####

####
# Reduction n64
SET(SRC_FILES
${COMMON_SRC_FILES}
reduction_n64.cpp
)
add_executable(nred64 ${SRC_FILES})
target_link_libraries(nred64 ${COMMON_LIBS} ${PVKERNEL_LIBRARIES})
####

