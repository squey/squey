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
