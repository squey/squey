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
core/squey_intrin.cpp
core/sse4detector.cpp
core/segfault_handler.cpp
core/serialize_numbers.cpp

core/PVArchive.cpp
core/PVArgument.cpp
core/PVOriginalAxisIndexType.cpp
core/PVAxisIndexType.cpp
core/PVAxisIndexCheckBoxType.cpp
core/PVBufferSlice.cpp
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
core/PVQueryBuilderJsonConverter.cpp
core/PVStreamingCompressor.cpp
core/PVVersion.cpp
)

SET(CORE_MOC_HEADERS
../include/pvkernel/core/PVProgressBox.h
)

set(OPENCL_SRC_FILES
	opencl/common.cpp
)

#########################################################################
# Ressources for pvkernel/filter
#########################################################################

SET(FILTER_SRC_FILES
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

SET(FILTER_MOC_HEADERS
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
rush/PVPipelineTask.cpp
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

../include/pvkernel/widgets/PVQueryBuilder.h
../include/pvkernel/widgets/PVLongLongSpinBox.h
../include/pvkernel/widgets/PVExportDlg.h
../include/pvkernel/widgets/PVCSVExporterWidget.h
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
widgets/PVQueryBuilder.cpp
widgets/PVLongLongSpinBox.cpp
widgets/PVFileDialog.cpp
widgets/PVExportDlg.cpp
widgets/PVMultipleFileDialog.cpp
)

SET(UI_FILES
widgets/PVColorDialog.ui
)

################################################
# Generate extra QT files (moc, uic, rcc)
################################################

set(SRC_FILES ${CORE_SRC_FILES} ${FILTER_SRC_FILES} ${RUSH_SRC_FILES} ${WIDGETS_SRC_FILES} ${OPENCL_SRC_FILES})
set(MOC_HEADERS ${CORE_MOC_HEADERS} ${FILTER_MOC_HEADERS} ${RUSH_MOC_HEADERS} ${WIDGETS_MOC_HEADERS})

QT6_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT6_WRAP_OPTIONS})
PV_QT6_WRAP_UI(UI_SRC pvkernel ${UI_FILES})

################################################
# Declare the new library and linkage options
################################################

# List files use to create the library
add_library(pvkernel SHARED ${SRC_FILES} ${MOC_SRC} ${UI_SRC})

set_target_properties(pvkernel PROPERTIES SOVERSION 1)

# Define third_party libs we have to link with for the library
set(QT6_LIBS Qt6Core Qt6OpenGL Qt6Widgets Qt6Gui Qt6Xml)
find_package(Qt6 COMPONENTS Core5Compat REQUIRED)

set(QT6_LIBS ${QT6_LIBS} Qt6WebEngineWidgets Qt6WebEngineCore)

set(PVKERNEL_LINK_LIBRARIES ${QT6_LIBS} ${ICU_LIBRARY} ${TBB_LIBRARIES}
	${OpenMP_LIBRARIES} ${Boost_LIBRARIES}
	${ARCHIVE_LIBRARIES} ${BREAKPAD_LIBRARIES} ${WINSOCK_LIB} ${HWLOC_LIBRARIES}
	${NUMA_LIBRARIES} ${SIGCPP_LIBRARIES} ${PVCOP_LIBRARIES} ${PVLOGGER_LIBRARIES} ${Python_LIBRARIES} Qt6::Core5Compat Arrow::arrow_shared Parquet::parquet_shared)

if(OPENCL_FOUND)
	list(APPEND PVKERNEL_LINK_LIBRARIES ${OPENCL_LIBRARIES})
endif(OPENCL_FOUND)

target_link_libraries(pvkernel ${PVKERNEL_LINK_LIBRARIES})


# Define installation directory
INSTALL(TARGETS pvkernel LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
			 RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
