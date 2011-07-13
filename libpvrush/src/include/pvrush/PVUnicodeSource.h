#ifndef PVRUSH_PVUNICODESOURCE_FILE_H
#define PVRUSH_PVUNICODESOURCE_FILE_H

#include <pvrush/PVRawSource.h>
#include <pvrush/PVChunkTransformUTF16.h>
#include <pvrush/PVChunkAlignUTF16Newline.h>

#include <QChar>

namespace PVRush {

template < template <class T> class Allocator = tbb::scalable_allocator >
class LibExport PVUnicodeSource : public PVRawSource<Allocator> {
public:
	typedef typename PVRawSource<Allocator>::alloc_chunk alloc_chunk;
public:
	PVUnicodeSource(PVInput_p input, size_t chunk_size, PVFilter::PVChunkFilter_f src_filter, const alloc_chunk &alloc = alloc_chunk()) :
		PVRawSource<Allocator>(input, _align, chunk_size, _utf16, src_filter, alloc)
	{
		//INIT_FILTER_NOPARAM(PVRush::PVUnicodeSource);
	}
public:
	bool discover()
	{
		return true;
	}
protected:
	PVChunkTransformUTF16 _utf16;
	PVChunkAlignUTF16Newline _align;

public:
//	CLASS_FILTER_NOPARAM_INPLACE(PVRush::PVUnicodeSource)
};

}

#endif
