/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVUNICODESOURCE_FILE_H
#define PVRUSH_PVUNICODESOURCE_FILE_H

#include <pvkernel/rush/PVRawSource.h>
#include <pvkernel/rush/PVChunkTransformUTF16.h>
#include <pvkernel/rush/PVChunkAlignUTF16Newline.h>

#include <QChar>

namespace PVRush {

template < template <class T> class Allocator = PVCore::PVMMapAllocator >
//template < template <class T> class Allocator = tbb::tbb_allocator >
class PVUnicodeSource : public PVRawSource<Allocator> {
public:
	typedef typename PVRawSource<Allocator>::alloc_chunk alloc_chunk;
public:
	PVUnicodeSource(PVInput_p input, size_t chunk_size, const alloc_chunk &alloc = alloc_chunk()) :
		PVRawSource<Allocator>(input, _align, chunk_size, _utf16, alloc)
	{
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
};

}

#endif
