# QT XML support

set(SRC_FILES
PVAxisFormat.cpp
pvdecoder.cpp
pvfile.cpp
PVFileRawSource.cpp
PVFormat.cpp
PVNormalizer.cpp
pvnormalizerfactory.cpp
PVNraw.cpp
PVNrawChild.cpp
PVNrawOutput.cpp
PVXmlParamParser.cpp
pvsource.cpp
pvstream.cpp
PVChunkAlign.cpp
PVChunkAlignUTF16Char.cpp
PVChunkAlignUTF16Newline.cpp
PVChunkAlignBoundary.cpp
PVInput.cpp
PVInputFile.cpp
PVInputPcap.cpp
PVOutput.cpp
PVOutputFile.cpp
PVCharsetDetect.cpp
PVChunkTransform.cpp
PVChunkTransformUTF16.cpp
PVAggregator.cpp
PVController.cpp
PVPipelineTask.cpp
PVControllerJob.cpp
PVControllerThread.cpp
PVExtractor.cpp
PVPluginsLoad.cpp
PVSourceCreatorFactory.cpp
PVTests.cpp
PVXmlTreeNodeDom.cpp

#uchardetect, from Mozilla's nsUniversalDetector
uchardetect/CharDistribution.cpp
uchardetect/JpCntx.cpp
uchardetect/LangBulgarianModel.cpp
uchardetect/LangCyrillicModel.cpp
uchardetect/LangGreekModel.cpp
uchardetect/LangHebrewModel.cpp
uchardetect/LangHungarianModel.cpp
uchardetect/LangThaiModel.cpp
uchardetect/nsBig5Prober.cpp
uchardetect/nsCharSetProber.cpp
uchardetect/nsEscCharsetProber.cpp
uchardetect/nsEscSM.cpp
uchardetect/nsEUCJPProber.cpp
uchardetect/nsEUCKRProber.cpp
uchardetect/nsEUCTWProber.cpp
uchardetect/nsGB2312Prober.cpp
uchardetect/nsHebrewProber.cpp
uchardetect/nsLatin1Prober.cpp
uchardetect/nsMBCSGroupProber.cpp
uchardetect/nsMBCSSM.cpp
uchardetect/nsSBCharSetProber.cpp
uchardetect/nsSBCSGroupProber.cpp
uchardetect/nsSJISProber.cpp
uchardetect/nsUniversalDetector.cpp
uchardetect/nsUTF8Prober.cpp
)

set(HEADER_FILES
include/pvrush/pv_axis_format.h
include/pvrush/pvdecoder.h
include/pvrush/PVFile.h
include/pvrush/pv_file_raw_source.h
include/pvrush/pv_format.h
include/pvrush/PVFormat.h
include/pvrush/PVNormalizer.h
include/pvrush/pvnormalizerfactory.h
include/pvrush/PVSource.h
include/pvrush/PVUnicodeSource.h
include/pvrush/PVCSVSource.h
include/pvrush/pvstream.h
include/pvrush/PVChunkAlign.h
include/pvrush/PVChunkAlignUTF16Char.h
include/pvrush/PVChunkAlignUTF16Newline.h
include/pvrush/PVChunkAlignBoundary.h
include/pvrush/PVInput.h
include/pvrush/PVInputFile.h
include/pvrush/PVInputPcap.h
include/pvrush/PVInputType.h
include/pvrush/PVNraw.h
include/pvrush/PVNrawChild.h
include/pvrush/PVNrawOutput.h
include/pvrush/PVOutputFile.h
include/pvrush/PVOutput.h
include/pvrush/PVPluginsLoad.h
include/pvrush/PVVrbAllocator.h
include/pvrush/PVRawSource.h
include/pvrush/PVSourceCreator.h
include/pvrush/PVCharsetDetect.h
include/pvrush/PVChunkTransform.h
include/pvrush/PVChunkTransformUTF16.h
include/pvrush/PVAggregator.h
include/pvrush/PVController.h
include/pvrush/PVControllerThread.h
include/pvrush/PVPipelineTask.h
include/pvrush/PVExtractor.h
include/pvrush/PVTests.h
include/pvrush/PVXmlParamParser.h
include/pvrush/PVXmlParamParserData.h

#uchardetect, from Mozilla's nsUniversalDetector
include/pvrush/uchardetect/CharDistribution.h
include/pvrush/uchardetect/JpCntx.h
include/pvrush/uchardetect/nsBig5Prober.h
include/pvrush/uchardetect/nsCharSetProber.h
include/pvrush/uchardetect/nsCodingStateMachine.h
include/pvrush/uchardetect/nscore.h
include/pvrush/uchardetect/nsEscCharsetProber.h
include/pvrush/uchardetect/nsEUCJPProber.h
include/pvrush/uchardetect/nsEUCKRProber.h
include/pvrush/uchardetect/nsEUCTWProber.h
include/pvrush/uchardetect/nsGB2312Prober.h
include/pvrush/uchardetect/nsHebrewProber.h
include/pvrush/uchardetect/nsLatin1Prober.h
include/pvrush/uchardetect/nsMBCSGroupProber.h
include/pvrush/uchardetect/nsPkgInt.h
include/pvrush/uchardetect/nsSBCharSetProber.h
include/pvrush/uchardetect/nsSBCSGroupProber.h
include/pvrush/uchardetect/nsSJISProber.h
include/pvrush/uchardetect/nsUniversalDetector.h
include/pvrush/uchardetect/nsUTF8Prober.h
include/pvrush/uchardetect/prmem.h
include/pvrush/uchardetect/universalchardet.h
include/pvrush/uchardetect/Big5Freq.tab
include/pvrush/uchardetect/EUCKRFreq.tab
include/pvrush/uchardetect/EUCTWFreq.tab
include/pvrush/uchardetect/GB2312Freq.tab
include/pvrush/uchardetect/JISFreq.tab
)

set(MOC_HEADERS
include/pvrush/PVControllerJob.h
include/pvrush/PVXmlTreeNodeDom.h
)
QT4_WRAP_CPP(MOC_SRC ${MOC_HEADERS})

add_library(pvrush SHARED ${SRC_FILES} ${MOC_SRC} ${HEADER_FILES} ${MOC_HEADERS})

set_target_properties(pvrush PROPERTIES SOVERSION 1)

if(WIN32)
	target_link_libraries(pvrush ${PVCORE_LIBRARY} ${PVFILTER_LIBRARY} ${QT_LIBRARIES} ${ARCHIVE_LIBRARY} ${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARY} ${Boost_LIBRARIES} ${DUMBNET_LIBRARIES} ${PCAP_LIBRARIES}  ${ICU_LIBRARY})
else(WIN32)
	target_link_libraries(pvrush ${PVCORE_LIBRARY} ${PVFILTER_LIBRARY} ${QT_LIBRARIES} ${ARCHIVE_LIBRARY} ${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARY} ${Boost_LIBRARIES} ${DUMBNET_LIBRARIES} ${PCAP_LIBRARIES}  ${ICU_LIBRARY} "rt" ${OpenMP_LIBRARIES})
endif(WIN32)

INSTALL(TARGETS pvrush DESTINATION .)
