/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVUNICODESOURCE_FILE_H
#define PVRUSH_PVUNICODESOURCE_FILE_H

#include <pvkernel/rush/PVRawSource.h>

namespace PVRush {

template < template <class T> class Allocator = PVCore::PVMMapAllocator >

class PVUnicodeSource : public PVRawSource<Allocator> {
public:
	typedef typename PVRawSource<Allocator>::alloc_chunk alloc_chunk;
public:
	PVUnicodeSource(PVInput_p input, size_t chunk_size, const alloc_chunk &alloc = alloc_chunk()) :
		PVRawSource<Allocator>(input, chunk_size, alloc)
	{
	}
public:
	bool discover() { return true; }
};

}

#endif
