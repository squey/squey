#
# @file
#

###############################################################################
# Define ressources
###############################################################################

set(SRC_FILES

common.cpp

PVAxesCombination.cpp
PVAxis.cpp
PVLayer.cpp
PVLayerFilter.cpp
PVLayerStack.cpp
PVLinesProperties.cpp
PVMapped.cpp
PVMappingProperties.cpp
PVScaled.cpp
PVScaledNrawCache.cpp
PVScalingProperties.cpp
PVRoot.cpp
PVScene.cpp
PVSource.cpp
PVStateMachine.cpp
PVView.cpp
PVCorrelationEngine.cpp
PVRangeSubSampler.cpp
PVDuckDBQuery.cpp
PVDuckDBQuerySource.cpp

plugins.cpp
)

if (PYTHON_SUPPORT)
	set(SRC_FILES ${SRC_FILES}
	PVPythonInterpreter.cpp
	PVPythonSource.cpp
	PVPythonSelection.cpp
	PVPythonSqlResult.cpp
	PVPythonInputDialog.cpp
	)
	set(MOC_HEADERS
	../include/squey/PVPythonInterpreter.h
	)
endif()

set(WIDGETS_MOC_HEADERS
../include/squey/widgets/PVAxisComboBox.h
../include/squey/widgets/PVMappingScalingEditDialog.h
../include/squey/widgets/PVNewLayerDialog.h

../include/squey/widgets/editors/PVAxisIndexCheckBoxEditor.h
../include/squey/widgets/editors/PVAxisIndexEditor.h
../include/squey/widgets/editors/PVLayerEnumEditor.h
../include/squey/widgets/editors/PVOriginalAxisIndexEditor.h
../include/squey/widgets/editors/PVViewRowsSpinBoxEditor.h
../include/squey/widgets/editors/PVZoneIndexEditor.h
)

set(WIDGETS_SRC_FILES
widgets/PVArgumentListWidgetFactory.cpp
widgets/PVAxisComboBox.cpp
widgets/PVAxisTypeWidget.cpp
widgets/PVMappingModeWidget.cpp
widgets/PVMappingScalingEditDialog.cpp
widgets/PVNewLayerDialog.cpp
widgets/PVScalingModeWidget.cpp

widgets/editors/PVAxisIndexCheckBoxEditor.cpp
widgets/editors/PVAxisIndexEditor.cpp
widgets/editors/PVLayerEnumEditor.cpp
widgets/editors/PVOriginalAxisIndexEditor.cpp
widgets/editors/PVViewRowsSpinBoxEditor.cpp
widgets/editors/PVZoneIndexEditor.cpp
)

###############################################################################
# Generate Qt files
###############################################################################
qt6_wrap_cpp(MOC_SRC ${MOC_HEADERS} ${WIDGETS_MOC_HEADERS} OPTIONS)

###############################################################################
# Define libs
###############################################################################

add_library(pvsquey SHARED ${SRC_FILES} ${WIDGETS_SRC_FILES} ${MOC_SRC})
target_compile_definitions(pvsquey PRIVATE PVSQUEY_LIBRARY)
set_target_properties(pvsquey PROPERTIES SOVERSION 1)
squey_speedup_compilation(pvsquey QT)

# DuckDB's profiling_utils.hpp holds a unique_ptr<ActiveTimer> member in a class
# whose constructor is defined inline, while ActiveTimer is only defined further
# down the same header. Clang instantiates that destructor eagerly in C++23 and
# rejects the incomplete type; below that it accepts it. The pimpl in
# PVDuckDBQuery keeps duckdb.hpp out of every other translation unit, so this
# single file is compiled against an older standard -- its public interface only
# exposes std:: and Squey types, which are ABI-compatible across both.
#
# C++20 and not older: pvcop's headers reach for std::countr_zero and the rest
# of <bit>. So the floor is C++20 and the ceiling is C++23, which leaves exactly
# one standard this file can be built with.
#
# It must stay out of the jumbo unit and of the shared PCH for that to hold: a
# per-file standard has no effect once the file is concatenated into a unit
# compiled as C++26. Set after squey_speedup_compilation(), which turns the
# unity build on.
# CXX_STANDARD is a target property only, so the standard is overridden through
# a compile option: it lands after the target-wide -std and therefore wins.
set_source_files_properties(PVDuckDBQuery.cpp PROPERTIES
	COMPILE_OPTIONS "-std=c++20"
	SKIP_UNITY_BUILD_INCLUSION ON
	SKIP_PRECOMPILE_HEADERS ON)

target_include_directories(pvsquey PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

# DuckDB is PRIVATE: the pimpl keeps duckdb.hpp out of the public headers.
target_link_libraries(pvsquey
	PUBLIC
		pvkernel
		Qt6::Core Qt6::Xml Qt6::Gui Qt6::Widgets
		${TBB_LIBRARIES} ${OpenMP_LIBRARIES} ${SIGCPP_LIBRARIES}
	PRIVATE
		${Boost_LIBRARIES} ${ICU_LIBRARY} ${DUCKDB_LIBRARIES})

###############################################################################
# Install files
###############################################################################

install(TARGETS pvsquey LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
				RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})
