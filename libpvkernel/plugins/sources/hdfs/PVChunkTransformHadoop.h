#ifndef PVCHUNKTRANSFORMHADOOP_FILE_H
#define PVCHUNKTRANSFORMHADOOP_FILE_H

#include <pvkernel/rush/PVChunkTransform.h>

namespace PVRush {

// We use a custom transform to allocate twice memory as what we need,
// so that there will be enough room for the UTF8->UTF16 conversion
class PVChunkTransformHadoop
{
public:
	virtual size_t next_read_size(size_t org_size) const { return org_size/2 ; }
	virtual size_t operator()(char* /*data*/, size_t len_read, size_t /*len_avail*/) const { return len_read; }
};

}

#endif
