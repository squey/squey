#ifndef PVRUSH_PVCSVSOURCE_FILE_H
#define PVRUSH_PVCSVSOURCE_FILE_H

#include <pvrush/PVUnicodeSource.h>
#include <pvrush/PVChunkTransformUTF16.h>
#include <pvfilter/PVFieldSplitterCSV.h>
#include <boost/bind.hpp>
#include <QChar>

namespace PVRush {

/*template < template <class T> class Allocator = std::allocator >
class LibExport PVCSVSource : public PVUnicodeSource<Allocator> {
public:
	typedef typename PVUnicodeSource<Allocator>::alloc_chunk alloc_chunk;
public:
	PVCSVSource(PVInput &input, size_t chunk_size, PVFilter::PVChunkFilter_f src_filter, const alloc_chunk &alloc = alloc_chunk()) :
		_org_src_filter(src_filter),
		PVUnicodeSource<Allocator>(input, chunk_size, src_filter, alloc_chunk)
	{
		//INIT_FILTER_NOPARAM(PVCSVSource<Allocator>);

		// The CSV splitter must be the first one, and then let's rock and roll !
		this->_src_filter = boost::bind(_org_src_filter, boost::bind(_f_csv, _1));
	}

public:
	bool discover()
	{
		return true;
	}
protected:
	PVChunkTransformUTF16 _utf16;
	PVFilter::PVFieldSplitterCSV _f_csv;
	PVFilter::PVChunkFilter_f _org_src_filter;

	//CLASS_FILTER_INPLACE(PVCSVSource<Allocator>)
};*/

}

#endif
