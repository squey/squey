#ifndef PVCHUNKUTF16FILTER_FILE_H
#define PVCHUNKUTF16FILTER_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVCharsetDetect.h>
#include <pvkernel/rush/PVChunkTransform.h>

#include <QTextCodec>

extern "C" {
#include <unicode/ucsdet.h>
#include <unicode/ucnv.h>
}

//#define CHUNKTRANSFORM_USE_QT_UNICODE

namespace PVRush {

/*! \brief Transform a chunk's data into UTF16
 * It will automatically detect the source's encoding.
 */
class LibKernelDecl PVChunkTransformUTF16 : public PVChunkTransform {
public:
	PVChunkTransformUTF16();
	virtual ~PVChunkTransformUTF16();
public:
	virtual size_t next_read_size(size_t org_size) const;
	/*! \brief Transform the given data into UTF16
	 *
	 * \todo Use ICU for conversion and detection, and see if this is faster than Qt !
	 */
	virtual size_t operator()(char* data, size_t len_read, size_t len_avail) const;
protected:
	mutable PVCharsetDetect _cd;
	/*! \brief Stores by how much the chunk size must be multiplied
	 */
	mutable float _mul_rs;
#ifdef CHUNKTRANSFORM_USE_QT_UNICODE
	mutable QTextDecoder *_decoder;
#else
	mutable UConverter* _ucnv;
	mutable UCharsetDetector* _csd;
	mutable bool _cd_found;
	mutable UChar* _tmp_dest;
	mutable size_t _tmp_dest_size;
#endif
};

}

#endif
