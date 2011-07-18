#ifndef PVELEMENTDATA_FILE_H
#define PVELEMENTDATA_FILE_H

#include <QSharedData>
#include <tbb/tbb_allocator.h>

#include <list>
#include <utility>

namespace PVCore {

// AG: no need to "LibExport" this
class PVElementData: public QSharedData
{
public:
	bool _valid;
	list_fields _fields;
	PVChunk *_parent;
	PVElement *_elt;
	buf_list_t _reallocated_buffers; // buf_list_t defined in PVBufferSlice.h

	~PVElementData()
	{
		static tbb::tbb_allocator<char> alloc;
		buf_list_t::const_iterator it;
		for (it = _reallocated_buffers.begin(); it != _reallocated_buffers.end(); it++) {
			alloc.deallocate(it->first, it->second);
		}
	}
};

}

#endif
