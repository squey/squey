#
# \file CMakeLists.txt
#
# Copyright (C) Picviz Labs 2010-2012

set(CORE_SRC_FILES
core/debug.cpp
core/environ.cpp
core/network.cpp
core/picviz_intrin.cpp
core/picviz_trace.cpp
core/sse4detector.cpp
core/segfault_handler.cpp

core/PVAllocators.cpp
core/PVArchive.cpp
core/PVArgument.cpp
core/PVOriginalAxisIndexType.cpp
core/PVAxisIndexType.cpp
core/PVAxisIndexCheckBoxType.cpp
core/PVBinaryOperation.cpp
core/PVBitCount.cpp
core/PVByteVisitor.cpp
core/PVBufferSlice.cpp
core/PVChunk.cpp
core/PVColor.cpp
core/PVCompList.cpp
core/PVClassLibrary.cpp
core/PVDateTimeParser.cpp
core/PVDirectory.cpp
core/PVElement.cpp
core/PVField.cpp
core/PVFileSerialize.cpp
core/PVHardwareConcurrency.cpp
core/PVHSVColor.cpp
core/PVPredefinedHSVColors.cpp
core/PVLogger.cpp
core/PVMatrix.cpp
core/PVPCRESplitter.cpp
core/PVProgressBox.cpp
core/PVPluginPresets.cpp
core/PVSelBitField.cpp
core/PVSerializeArchive.cpp
core/PVSerializeArchiveFixError.cpp
core/PVSerializeArchiveOptions.cpp
core/PVSerializeArchiveZip.cpp
core/PVSerializeObject.cpp
core/PVSplitter.cpp
core/PVUtils.cpp
core/PVUnicodeString.cpp
core/PVUnicodeString16.cpp
core/PVVersion.cpp
core/PVAllocators.cpp
core/PVRecentItemsManager.cpp
core/string_tbb.cpp
core/PVConfig.cpp
core/PVFileHelper.cpp
core/PVMemory2D.cpp
)

if(PYTHONLIBS_FOUND)
	set(CORE_SRC_FILES
	core/PVPython.cpp
	${CORE_SRC_FILES}
	)
endif(PYTHONLIBS_FOUND)

SET(CORE_MOC_HEADERS
include/pvkernel/core/lambda_connect.h
include/pvkernel/core/PVProgressBox.h
)

set(CORE_HEADER_FILES
include/pvkernel/core/debug.h
include/pvkernel/core/environ.h
include/pvkernel/core/general.h
include/pvkernel/core/network.h
include/pvkernel/core/picviz_intrin.h
include/pvkernel/core/sse4detector.h
include/pvkernel/core/pv_pcre_splitter.h
include/pvkernel/core/pv_splitter.h
include/pvkernel/core/PVStringUtils.h
include/pvkernel/core/hash_sharedptr.h

include/pvkernel/core/PVArchive.h
include/pvkernel/core/PVArgument.h
include/pvkernel/core/PVOriginalAxisIndexType.h
include/pvkernel/core/PVAxisIndexType.h
include/pvkernel/core/PVZoneIndexType.h
include/pvkernel/core/PVAxesIndexType.h
include/pvkernel/core/PVBitCount.h
include/pvkernel/core/PVColor.h
include/pvkernel/core/PVClassLibrary.h
include/pvkernel/core/PVDateTimeParser.h
include/pvkernel/core/PVDirectory.h
include/pvkernel/core/PVEnumType.h
include/pvkernel/core/PVFileSerialize.h
include/pvkernel/core/PVColorGradientDualSliderType.h
include/pvkernel/core/PVBufferSlice.h
include/pvkernel/core/PVChunk.h
include/pvkernel/core/PVElement.h
include/pvkernel/core/PVField.h
include/pvkernel/core/PVHSVColor.h
include/pvkernel/core/PVListFastCmp.h
include/pvkernel/core/PVLogger.h
include/pvkernel/core/PVMeanValue.h
include/pvkernel/core/PVPlainTextType.h
include/pvkernel/core/PVPluginPresets.h
include/pvkernel/core/PVRegistrableClass.h
include/pvkernel/core/PVSerializeArchive.h
include/pvkernel/core/PVSerializeArchiveFixError.h
include/pvkernel/core/PVSerializeArchiveOptions.h
include/pvkernel/core/PVSerializeArchiveZip.h
include/pvkernel/core/PVSerializeObject.h
include/pvkernel/core/PVSpinBoxType.h
include/pvkernel/core/PVTBBMaxArray.h
include/pvkernel/core/PVTBBMinMaxArray.h
include/pvkernel/core/PVTextEditType.h
include/pvkernel/core/PVTimeFormatType.h
include/pvkernel/core/PVTypeInfo.h
include/pvkernel/core/PVUtils.h
include/pvkernel/core/PVUnicodeString.h
include/pvkernel/core/PVVersion.h
include/pvkernel/core/PVRecentItemsManager.h
include/pvkernel/core/PVConfig.h
include/pvkernel/core/PVVector.h
include/pvkernel/core/PVFileHelper.h
include/pvkernel/core/PVMemory2D.h
include/pvkernel/core/PVColumnIndexes.h

include/pvkernel/core/PVPercentRangeType.h
)

if(PYTHONLIBS_FOUND)
	set(CORE_HEADER_FILES
	${CORE_HEADER_FILES}
	include/pvkernel/core/PVPython.h
	)
endif(PYTHONLIBS_FOUND)

if (CUDA_FOUND)
	set(CUDA_SRC_FILES
		cuda/common.cpp
	)

	set(CUDA_HEADER_FILES
		include/pvkernel/cuda/common.h
	)
else(CUDA_FOUND)
	set(CUDA_SRC_FILES "")
	set(CUDA_HEADER_FILES "")
endif(CUDA_FOUND)

SET(FILTER_SRC_FILES
filter/PVChunkFilter.cpp
filter/PVChunkFilterByElt.cpp
filter/PVChunkFilterByEltCancellable.cpp
filter/PVChunkFilterByEltRestoreInvalid.cpp
filter/PVChunkFilterByEltSaveInvalid.cpp
filter/PVChunkFilterSource.cpp
filter/PVChunkFilterCountElts.cpp
filter/PVChunkFilterDumpElts.cpp
filter/PVElementFilterByFields.cpp
filter/PVElementFilterRandInvalid.cpp
filter/PVFieldFilterGrep.cpp
filter/PVFieldsFilter.cpp
filter/PVFieldsMappingFilter.cpp
filter/PVFieldSplitterChar.cpp
filter/PVFieldSplitterChunkMatch.cpp
filter/PVFieldSplitterUTF16Char.cpp
filter/PVFieldsFilterParamWidget.cpp
filter/PVPureMappingProcessing.cpp
filter/PVPluginsLoad.cpp
filter/PVSeqChunkFunction.cpp
)

SET(FILTER_HEADER_FILES
include/pvkernel/filter/PVChunkFilter.h
include/pvkernel/filter/PVChunkFilterByElt.h
include/pvkernel/filter/PVChunkFilterByEltCancellable.h
include/pvkernel/filter/PVChunkFilterByEltRestoreInvalid.h
include/pvkernel/filter/PVChunkFilterByEltSaveInvalid.h
include/pvkernel/filter/PVChunkFilterSource.h
include/pvkernel/filter/PVChunkFilterCountElts.h
include/pvkernel/filter/PVChunkFilterDumpElts.h
include/pvkernel/filter/PVElementFilterByFields.h
include/pvkernel/filter/PVElementFilter.h
include/pvkernel/filter/PVFieldsFilter.h
include/pvkernel/filter/PVFilterFunction.h
include/pvkernel/filter/PVElementFilterRandInvalid.h
include/pvkernel/filter/PVFieldFilterGrep.h
include/pvkernel/filter/PVFieldSplitterChar.h
include/pvkernel/filter/PVFieldSplitterChunkMatch.h
include/pvkernel/filter/PVFieldSplitterUTF16Char.h
include/pvkernel/filter/PVFieldsMappingFilter.h
include/pvkernel/filter/PVPluginsLoad.h
include/pvkernel/filter/PVPureMappingProcessing.h
include/pvkernel/filter/PVSeqChunkFunction.h
)

SET(FILTER_MOC_HEADERS
include/pvkernel/filter/PVFieldsFilterParamWidget.h
)

set(RUSH_SRC_FILES
rush/PVAxisFormat.cpp
rush/pvdecoder.cpp
rush/PVFileRawSource.cpp
rush/PVFormat.cpp
rush/PVNraw.cpp
rush/PVNrawDiskBackend.cpp
rush/PVNrawChild.cpp
rush/PVNrawOutput.cpp
rush/PVNormalizer.cpp
rush/PVXmlParamParser.cpp
rush/PVChunkAlign.cpp
rush/PVChunkAlignUTF16Char.cpp
rush/PVChunkAlignUTF16Newline.cpp
rush/PVChunkAlignBoundary.cpp
rush/PVFormatVersion.cpp
rush/PVInput.cpp
rush/PVInputFile.cpp
rush/PVInputPcap.cpp
rush/PVOutput.cpp
rush/PVOutputFile.cpp
rush/PVCharsetDetect.cpp
rush/PVChunkTransform.cpp
rush/PVChunkTransformUTF16.cpp
rush/PVAggregator.cpp
rush/PVController.cpp
rush/PVPipelineTask.cpp
rush/PVControllerJob.cpp
rush/PVControllerThread.cpp
rush/PVExtractor.cpp
rush/PVPluginsLoad.cpp
rush/PVSourceCreatorFactory.cpp
rush/PVTags.cpp
rush/PVTests.cpp
rush/PVUtils.cpp
rush/PVXmlTreeNodeDom.cpp
rush/PVRawSourceBase.cpp
rush/PVSourceDescription.cpp
rush/PVNrawCacheManager.cpp

#uchardetect, from Mozilla's nsUniversalDetector
rush/uchardetect/CharDistribution.cpp
rush/uchardetect/JpCntx.cpp
rush/uchardetect/LangBulgarianModel.cpp
rush/uchardetect/LangCyrillicModel.cpp
rush/uchardetect/LangGreekModel.cpp
rush/uchardetect/LangHebrewModel.cpp
rush/uchardetect/LangHungarianModel.cpp
rush/uchardetect/LangThaiModel.cpp
rush/uchardetect/nsBig5Prober.cpp
rush/uchardetect/nsCharSetProber.cpp
rush/uchardetect/nsEscCharsetProber.cpp
rush/uchardetect/nsEscSM.cpp
rush/uchardetect/nsEUCJPProber.cpp
rush/uchardetect/nsEUCKRProber.cpp
rush/uchardetect/nsEUCTWProber.cpp
rush/uchardetect/nsGB2312Prober.cpp
rush/uchardetect/nsHebrewProber.cpp
rush/uchardetect/nsLatin1Prober.cpp
rush/uchardetect/nsMBCSGroupProber.cpp
rush/uchardetect/nsMBCSSM.cpp
rush/uchardetect/nsSBCharSetProber.cpp
rush/uchardetect/nsSBCSGroupProber.cpp
rush/uchardetect/nsSJISProber.cpp
rush/uchardetect/nsUniversalDetector.cpp
rush/uchardetect/nsUTF8Prober.cpp
)

if(PYTHONLIBS_FOUND)
	set(RUSH_SRC_FILES
	${RUSH_SRC_FILES}
	#rush/PVNrawPython.cpp
)

set(RUSH_HEADER_FILES
include/pvkernel/rush/PVAxisFormat.h
include/pvkernel/rush/PVAggregator.h
include/pvkernel/rush/PVController.h
include/pvkernel/rush/PVControllerThread.h
include/pvkernel/rush/pvdecoder.h
include/pvkernel/rush/PVFile.h
include/pvkernel/rush/pv_file_raw_source.h
include/pvkernel/rush/pv_format.h
include/pvkernel/rush/PVFormat.h
include/pvkernel/rush/PVSource.h
include/pvkernel/rush/PVUnicodeSource.h
include/pvkernel/rush/PVCSVSource.h
include/pvkernel/rush/pvstream.h
include/pvkernel/rush/PVChunkAlign.h
include/pvkernel/rush/PVChunkAlignUTF16Char.h
include/pvkernel/rush/PVChunkAlignUTF16Newline.h
include/pvkernel/rush/PVChunkAlignBoundary.h
include/pvkernel/rush/PVCharsetDetect.h
include/pvkernel/rush/PVChunkTransform.h
include/pvkernel/rush/PVChunkTransformUTF16.h
include/pvkernel/rush/PVFormatVersion.h
include/pvkernel/rush/PVExtractor.h
include/pvkernel/rush/PVInput.h
include/pvkernel/rush/PVInputFile.h
include/pvkernel/rush/PVInputPcap.h
include/pvkernel/rush/PVNraw.h
include/pvkernel/rush/PVNrawDiskBackend.h
include/pvkernel/rush/PVNrawChild.h
include/pvkernel/rush/PVNrawOutput.h
include/pvkernel/rush/PVOutputFile.h
include/pvkernel/rush/PVOutput.h
include/pvkernel/rush/PVPluginsLoad.h
include/pvkernel/rush/PVVrbAllocator.h
include/pvkernel/rush/PVNormalizer.h
include/pvkernel/rush/PVRawSource.h
include/pvkernel/rush/PVRawSourceBase.h
include/pvkernel/rush/PVSourceCreator.h
include/pvkernel/rush/PVPipelineTask.h
include/pvkernel/rush/PVTags.h
include/pvkernel/rush/PVTests.h
include/pvkernel/rush/PVUtils.h
include/pvkernel/rush/PVXmlParamParser.h
include/pvkernel/rush/PVXmlParamParserData.h
include/pvkernel/rush/PVSourceDescription.h
include/pvkernel/rush/PVNrawCacheManager.h

#uchardetect, from Mozilla's nsUniversalDetector
include/pvkernel/rush/uchardetect/CharDistribution.h
include/pvkernel/rush/uchardetect/JpCntx.h
include/pvkernel/rush/uchardetect/nsBig5Prober.h
include/pvkernel/rush/uchardetect/nsCharSetProber.h
include/pvkernel/rush/uchardetect/nsCodingStateMachine.h
include/pvkernel/rush/uchardetect/nscore.h
include/pvkernel/rush/uchardetect/nsEscCharsetProber.h
include/pvkernel/rush/uchardetect/nsEUCJPProber.h
include/pvkernel/rush/uchardetect/nsEUCKRProber.h
include/pvkernel/rush/uchardetect/nsEUCTWProber.h
include/pvkernel/rush/uchardetect/nsGB2312Prober.h
include/pvkernel/rush/uchardetect/nsHebrewProber.h
include/pvkernel/rush/uchardetect/nsLatin1Prober.h
include/pvkernel/rush/uchardetect/nsMBCSGroupProber.h
include/pvkernel/rush/uchardetect/nsPkgInt.h
include/pvkernel/rush/uchardetect/nsSBCharSetProber.h
include/pvkernel/rush/uchardetect/nsSBCSGroupProber.h
include/pvkernel/rush/uchardetect/nsSJISProber.h
include/pvkernel/rush/uchardetect/nsUniversalDetector.h
include/pvkernel/rush/uchardetect/nsUTF8Prober.h
include/pvkernel/rush/uchardetect/prmem.h
include/pvkernel/rush/uchardetect/universalchardet.h
include/pvkernel/rush/uchardetect/Big5Freq.tab
include/pvkernel/rush/uchardetect/EUCKRFreq.tab
include/pvkernel/rush/uchardetect/EUCTWFreq.tab
include/pvkernel/rush/uchardetect/GB2312Freq.tab
include/pvkernel/rush/uchardetect/JISFreq.tab
)

set(RUSH_MOC_HEADERS
include/pvkernel/rush/PVControllerJob.h
include/pvkernel/rush/PVXmlTreeNodeDom.h
include/pvkernel/rush/PVInputType.h
)

endif(PYTHONLIBS_FOUND)
#
# Widgets
#
set(WIDGETS_MOC_HEADERS
include/pvkernel/widgets/qkeysequencewidget.h
include/pvkernel/widgets/qkeysequencewidget_p.h
include/pvkernel/widgets/PVAbstractDataTreeMaskProxyModel.h
include/pvkernel/widgets/PVArgumentListWidget.h
include/pvkernel/widgets/PVColorDialog.h
include/pvkernel/widgets/PVColorPicker.h
include/pvkernel/widgets/PVPresetsWidget.h
include/pvkernel/widgets/PVGraphicsView.h
include/pvkernel/widgets/PVGraphicsViewInteractor.h

include/pvkernel/widgets/PVConfigPopupWidget.h

include/pvkernel/widgets/PVAbstractRangePicker.h

include/pvkernel/widgets/editors/PVColorGradientDualSliderEditor.h
include/pvkernel/widgets/editors/PVEnumEditor.h
include/pvkernel/widgets/editors/PVPlainTextEditor.h
include/pvkernel/widgets/editors/PVRegexpEditor.h
include/pvkernel/widgets/editors/PVTimeFormatEditor.h

include/pvkernel/widgets/editors/PVPercentRangeEditor.h
)

set(WIDGETS_HEADER_FILES
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
widgets/PVDataTreeModel.cpp
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
widgets/PVConfigPopupWidget.cpp
widgets/PVTextPopupWidget.cpp

widgets/PVHelpWidget.cpp

widgets/PVAbstractRangePicker.cpp
)

SET(UI_FILES
widgets/PVColorDialog.ui
)

if(PYTHONLIBS_FOUND)
	include_directories(BEFORE ${PYTHON_INCLUDE_DIRS})
endif(PYTHONLIBS_FOUND)

set(SRC_FILES ${CORE_SRC_FILES} ${CUDA_SRC_FILES} ${FILTER_SRC_FILES} ${RUSH_SRC_FILES} ${WIDGETS_SRC_FILES})
set(HEADER_FILES ${CORE_HEADER_FILES} ${CUDA_HEADER_FILES} ${FILTER_HEADER_FILES} ${RUSH_HEADER_FILES} ${WIDGETS_HEADER_FILES})
set(MOC_HEADERS ${CORE_MOC_HEADERS} ${FILTER_MOC_HEADERS} ${RUSH_MOC_HEADERS} ${WIDGETS_MOC_HEADERS})

QT4_WRAP_CPP(MOC_SRC ${MOC_HEADERS} OPTIONS ${QT4_WRAP_OPTIONS})
PV_QT4_WRAP_UI(UI_SRC ${UI_FILES})

add_library(pvkernel SHARED ${SRC_FILES} ${HEADER_FILES} ${MOC_SRC} ${MOC_HEADERS} ${UI_FILES})

set_target_properties(pvkernel PROPERTIES SOVERSION 1)

set(PVKERNEL_LINK_LIBRARIES ${QT_LIBRARIES} ${ICU_LIBRARY} ${TBB_LIBRARIES} ${OpenMP_LIBRARIES} ${PYTHON_LIBRARY} ${Boost_LIBRARIES_PYTHON} ${ARCHIVE_LIBRARY} ${PCAP_LIBRARIES} ${WINSOCK_LIB} ${X11_LIBRARIES} ${HWLOC_LIBRARIES} ${NUMA_LIBRARIES})

if(CUDA_FOUND)
	set(PVKERNEL_LINK_LIBRARIES ${PVKERNEL_LINK_LIBRARIES} ${CUDA_LIBRARIES})
endif(CUDA_FOUND)	
#if(PYTHONLIBS_FOUND)
#	set(PVKERNEL_LINK_LIBRARIES ${PVKERNEL_LINK_LIBRARIES} ${PYTHON_LIBRARY} ${Boost_LIBRARIES_PYTHON})
#endif(PYTHONLIBS_FOUND)
target_link_libraries(pvkernel ${PVKERNEL_LINK_LIBRARIES})

INSTALL(TARGETS pvkernel LIBRARY DESTINATION .
			 RUNTIME DESTINATION .)
