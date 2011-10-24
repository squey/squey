set(SAD_SRC_FILES
init.cpp
)

set(SAD_HEADER_FILES
include/sad/export.h
include/sad/general.h
include/sad/init.h
)

SET(SRC_FILES ${SAD_SRC_FILES})
SET(HEADER_FILES ${SAD_HEADER_FILES})

add_library(sad SHARED ${SRC_FILES} ${HEADER_FILES})

set_target_properties(sad PROPERTIES SOVERSION 1)

target_link_libraries(sad ${QT_LIBRARIES} ${Boost_LIBRARIES})

install(TARGETS sad LIBRARY DESTINATION .
		    RUNTIME DESTINATION .)
