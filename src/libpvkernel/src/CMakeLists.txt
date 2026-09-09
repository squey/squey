#
# @file
#

###############################################################################
# Ressource declaration
###############################################################################

#########################################################################
# Ressources for pvkernel/core
#########################################################################
set(CORE_SRC_FILES
core/network.cpp

core/PVArchive.cpp
core/PVArgument.cpp
core/PVOriginalAxisIndexType.cpp
core/PVAxisIndexType.cpp
core/PVAxisIndexCheckBoxType.cpp
core/PVBufferSlice.cpp
core/PVCpuFeatures.cpp
core/PVTextChunk.cpp
core/PVColor.cpp
core/PVCompList.cpp
core/PVClassLibrary.cpp
core/PVDateTimeParser.cpp
core/PVDirectory.cpp
core/PVElement.cpp
core/PVField.cpp
core/PVHSVColor.cpp
core/PVPredefinedHSVColors.cpp
core/PVLogger.cpp
core/PVProgressBox.cpp
core/PVPluginPresets.cpp
core/PVSelBitField.cpp
core/PVSerializeArchive.cpp
core/PVSerializeArchiveZip.cpp
core/PVSerializeObject.cpp
core/PVUtils.cpp
core/PVAllocators.cpp
core/PVRecentItemsManager.cpp
core/PVConfig.cpp
core/PVFileHelper.cpp
core/PVMemory2D.cpp
core/PVStreamingCompressor.cpp
core/PVVersion.cpp
core/PVTheme.cpp
core/PVSingleton.cpp
)

# Crashpad supports all three targets, unlike the Breakpad setup it replaced
# which was Linux only.
set(CORE_SRC_FILES ${CORE_SRC_FILES}
	core/segfault_handler.cpp
)

set(CORE_MOC_HEADERS
../include/pvkernel/core/PVProgressBox.h
../include/pvkernel/core/PVDBusConnection.h
../include/pvkernel/core/PVTheme.h
)

set(OPENCL_SRC_FILES
	opencl/common.cpp
)

#########################################################################
# Ressources for pvkernel/filter
#########################################################################

set(FILTER_SRC_FILES
filter/PVChunkFilterByElt.cpp
filter/PVChunkFilterByEltCancellable.cpp
filter/PVChunkFilterDumpElts.cpp
filter/PVChunkFilterRemoveInvalidElts.cpp
filter/PVElementFilterByFields.cpp
filter/PVElementFilterByAxes.cpp
filter/PVFieldFilterGrep.cpp
filter/PVFieldsFilter.cpp
filter/PVFieldsMappingFilter.cpp
filter/PVFieldSplitterChunkMatch.cpp
filter/PVPluginsLoad.cpp
)

set(FILTER_MOC_HEADERS
../include/pvkernel/filter/PVFieldsFilterParamWidget.h
)

#########################################################################
# Ressources for pvkernel/rush
#########################################################################

set(RUSH_SRC_FILES
rush/PVAxisFormat.cpp
rush/PVFormat.cpp
rush/PVNraw.cpp
rush/PVNrawOutput.cpp
rush/PVTypesDiscoveryOutput.cpp
rush/PVNormalizer.cpp
rush/PVXmlParamParser.cpp
rush/PVFormatVersion.cpp
rush/PVInputFile.cpp
rush/PVCharsetDetect.cpp
rush/PVAggregator.cpp
rush/PVControllerJob.cpp
rush/PVExtractor.cpp
rush/PVPluginsLoad.cpp
rush/PVSourceCreatorFactory.cpp
rush/PVTests.cpp
rush/PVUtils.cpp
rush/PVXmlTreeNodeDom.cpp
rush/PVRawSourceBase.cpp
rush/PVSourceDescription.cpp
rush/PVNrawCacheManager.cpp
rush/PVCSVExporter.cpp

#uchardetect, from Mozilla's nsUniversalDetector
third_party/uchardetect/CharDistribution.cpp
third_party/uchardetect/JpCntx.cpp
third_party/uchardetect/nsBig5Prober.cpp
third_party/uchardetect/nsCharSetProber.cpp
third_party/uchardetect/nsEscCharsetProber.cpp
third_party/uchardetect/nsEscSM.cpp
third_party/uchardetect/nsEUCJPProber.cpp
third_party/uchardetect/nsEUCKRProber.cpp
third_party/uchardetect/nsEUCTWProber.cpp
third_party/uchardetect/nsGB2312Prober.cpp
third_party/uchardetect/nsHebrewProber.cpp
third_party/uchardetect/nsLatin1Prober.cpp
third_party/uchardetect/nsMBCSGroupProber.cpp
third_party/uchardetect/nsMBCSSM.cpp
third_party/uchardetect/nsSBCharSetProber.cpp
third_party/uchardetect/nsSBCSGroupProber.cpp
third_party/uchardetect/nsSJISProber.cpp
third_party/uchardetect/nsUniversalDetector.cpp
third_party/uchardetect/nsUTF8Prober.cpp
third_party/uchardetect/uchardet.cpp
third_party/uchardetect/LangModels/LangArabicModel.cpp
third_party/uchardetect/LangModels/LangBulgarianModel.cpp
third_party/uchardetect/LangModels/LangDanishModel.cpp
third_party/uchardetect/LangModels/LangEsperantoModel.cpp
third_party/uchardetect/LangModels/LangFrenchModel.cpp
third_party/uchardetect/LangModels/LangGermanModel.cpp
third_party/uchardetect/LangModels/LangGreekModel.cpp
third_party/uchardetect/LangModels/LangHebrewModel.cpp
third_party/uchardetect/LangModels/LangHungarianModel.cpp
third_party/uchardetect/LangModels/LangRussianModel.cpp
third_party/uchardetect/LangModels/LangSpanishModel.cpp
third_party/uchardetect/LangModels/LangThaiModel.cpp
third_party/uchardetect/LangModels/LangTurkishModel.cpp
third_party/uchardetect/LangModels/LangVietnameseModel.cpp
)

set(RUSH_MOC_HEADERS
../include/pvkernel/rush/PVControllerJob.h
../include/pvkernel/rush/PVXmlTreeNodeDom.h
../include/pvkernel/rush/PVInputType.h
)

#########################################################################
# Ressources for pvkernel/widgets
#########################################################################
set(WIDGETS_MOC_HEADERS
../include/pvkernel/widgets/qkeysequencewidget.h
../include/pvkernel/widgets/qkeysequencewidget_p.h
../include/pvkernel/widgets/PVArgumentListWidget.h
../include/pvkernel/widgets/PVColorDialog.h
../include/pvkernel/widgets/PVColorPicker.h
../include/pvkernel/widgets/PVPresetsWidget.h
../include/pvkernel/widgets/PVGraphicsView.h
../include/pvkernel/widgets/PVGraphicsViewInteractor.h

../include/pvkernel/widgets/PVAbstractRangePicker.h

../include/pvkernel/widgets/editors/PVColorGradientDualSliderEditor.h
../include/pvkernel/widgets/editors/PVEnumEditor.h
../include/pvkernel/widgets/editors/PVPlainTextEditor.h
../include/pvkernel/widgets/editors/PVRegexpEditor.h
../include/pvkernel/widgets/editors/PVTimeFormatEditor.h

../include/pvkernel/widgets/editors/PVPercentRangeEditor.h

../include/pvkernel/widgets/PVLongLongSpinBox.h
../include/pvkernel/widgets/PVExportDlg.h
../include/pvkernel/widgets/PVCSVExporterWidget.h
../include/pvkernel/widgets/PVModdedIcon.h
../include/pvkernel/widgets/PVFileDialog.h
)

set(WIDGETS_SRC_FILES
widgets/qkeysequencewidget/qkeysequencewidget.cpp
widgets/presetswidget/PVPresetsWidget.cpp
widgets/PVArgumentListModel.cpp
widgets/PVArgumentListWidget.cpp
widgets/PVArgumentListWidgetFactory.cpp
widgets/PVComboBox.cpp
widgets/PVColorDialog.cpp
widgets/PVColorPicker.cpp
widgets/PVGraphicsView.cpp
widgets/PVGraphicsViewInteractor.cpp
widgets/PVGraphicsViewInteractorScene.cpp

widgets/editors/PVColorGradientDualSliderEditor.cpp
widgets/editors/PVEnumEditor.cpp
widgets/editors/PVPlainTextEditor.cpp
widgets/editors/PVRegexpEditor.cpp
widgets/editors/PVTimeFormatEditor.cpp
widgets/editors/PVPercentRangeEditor.cpp
widgets/PVUtils.cpp

widgets/PVPopupWidget.cpp
widgets/PVTextPopupWidget.cpp
widgets/PVHelpWidget.cpp
widgets/PVAbstractRangePicker.cpp
widgets/PVLayerNamingPatternDialog.cpp
widgets/PVLongLongSpinBox.cpp
widgets/PVFileDialog.cpp
widgets/PVExportDlg.cpp
widgets/PVMultipleFileDialog.cpp
widgets/PVModdedIcon.cpp
)

set(UI_FILES
widgets/PVColorDialog.ui
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

set(SRC_FILES ${CORE_SRC_FILES} ${FILTER_SRC_FILES} ${RUSH_SRC_FILES} ${WIDGETS_SRC_FILES} ${OPENCL_SRC_FILES})
set(MOC_HEADERS ${CORE_MOC_HEADERS} ${FILTER_MOC_HEADERS} ${RUSH_MOC_HEADERS} ${WIDGETS_MOC_HEADERS})

qt6_wrap_cpp(MOC_SRC ${MOC_HEADERS} OPTIONS)
pv_qt6_wrap_ui(UI_SRC pvkernel ${UI_FILES})

################################################
# Declare the new library and linkage options
################################################

# List files use to create the library
add_library(pvkernel SHARED ${SRC_FILES} ${MOC_SRC} ${UI_SRC})
target_compile_definitions(pvkernel PRIVATE PVKERNEL_LIBRARY)
set_target_properties(pvkernel PROPERTIES SOVERSION 1)
# segfault_handler.cpp is kept out of the jumbo units: mini_chromium defines a
# CHECK macro of its own, which clashes with the one of squey_bench.h as soon as
# both land in the same translation unit.
squey_speedup_compilation(pvkernel QT SKIP_UNITY core/segfault_handler.cpp)

# The headers of the library, and the ones generated beside them (ui_*.h).
target_include_directories(pvkernel PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
# PUBLIC: PVCharsetDetect.h, a header of the library, includes <nscore.h> from
# the vendored uchardetect.
target_include_directories(pvkernel PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/uchardetect>)
target_include_directories(pvkernel SYSTEM PUBLIC ${CRASHPAD_INCLUDE_DIRS})

target_link_libraries(pvkernel
	PUBLIC
		squey_base
		pvcop pvlogger pvhwloc
		Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Xml Qt6::DBus Qt6::Core5Compat
		${TBB_LIBRARIES} ${OpenMP_LIBRARIES} ${Boost_LIBRARIES}
		${SIGCPP_LIBRARIES} ${ICU_LIBRARY}
		# These appear in the headers of the library, as inline code that its
		# users end up instantiating: they have to link them as well. The macOS
		# linker is the one that says so, the ELF one resolves them indirectly.
		Squey::OpenCL Python::Python CURL::libcurl
	PRIVATE
		${ARCHIVE_LIBRARIES} ${CRASHPAD_LIBRARIES})

# Define installation directory
install(TARGETS pvkernel LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
		RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})
