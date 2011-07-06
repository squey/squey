/*
 * $Id: PVChunk.h 3250 2011-07-05 12:31:00Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVCHUNK_FILE_H
#define PVCHUNK_FILE_H

#include <pvcore/general.h>
#include <pvcore/PVElement.h>

#include <tbb/tbb_allocator.h>

#ifdef WIN32
	#include <pvcore/win32-vs2008-stdint.h>
#else
	#include <stdint.h>
#endif

#include <memory>
#include <cassert>
#include <list>

namespace PVFilter
{
	class PVRawSourceBase;
};

namespace PVRush {
	class PVAggregator;
};

namespace PVCore {

typedef uint64_t chunk_index;
typedef std::list< PVElement, tbb::tbb_allocator<PVElement> > list_elts;

// Describe chunk interface with no allocator template
// Useful in order to use chunks as function arguments...
class LibExport PVChunk {
friend class PVRush::PVAggregator;

public:
	PVChunk() : _index(0), _n_elts_invalid(0) {};
	virtual ~PVChunk() {};
public:
	virtual char* begin() const = 0;
	char* end() const { return _logical_end; };
	char* physical_end() const { return _physical_end; };
	void set_end(char* p)
	{
		assert(p <= _physical_end);
		_logical_end = p;
	};
	chunk_index index() const { return _index; };
	chunk_index agg_index() const { return _agg_index; };
	chunk_index last_elt_index() const { return _index + _elts.size() - 1; };
	chunk_index last_elt_agg_index() const { return _agg_index + _elts.size() - 1; };
	inline void set_index(chunk_index i) { _index = i; };
	inline void set_elts_stat(size_t nelts_org, size_t nelts_valid) { _nelts_org = nelts_org; _nelts_valid = nelts_valid; }
	inline void get_elts_stat(size_t& nelts_org, size_t& nelts_valid) { nelts_org = _nelts_org; nelts_valid = _nelts_valid; }

	size_t size() const { return (size_t) ((uintptr_t)_logical_end - (uintptr_t)begin()); };
	size_t avail() const { return (size_t) ((uintptr_t)_physical_end - (uintptr_t)_logical_end); };

	virtual void free() = 0;
	virtual PVChunk* realloc_grow(size_t n) = 0;

	list_elts& elements() {return _elts;};
	list_elts const& c_elements() const {return _elts;}; 

	PVFilter::PVRawSourceBase* source() const { return _source; };
protected:
	// Useful datas
	char* _logical_end;
	char* _physical_end;
	list_elts _elts;
	chunk_index _index;
	chunk_index _agg_index;
	PVFilter::PVRawSourceBase *_source;
	PVRow _n_elts_invalid;
	size_t _nelts_org;
	size_t _nelts_valid;
};

template < template <class T> class Allocator = std::allocator >
class LibExport PVChunkMem : public PVChunk {
public:
	typedef Allocator<char> alloc_chunk;
private:
	PVChunkMem(alloc_chunk const& a) :
		PVChunk(), _alloc(a)
	{
	};

	virtual ~PVChunkMem() {}
public:
    char* begin() const { return (char*)(this+1); };
	static PVChunkMem* allocate(size_t size, PVFilter::PVRawSourceBase* parent, alloc_chunk a = alloc_chunk())
	{
		size_t size_alloc = sizeof(PVChunkMem<Allocator>)+size+1;
		PVChunkMem<Allocator>* p = (PVChunkMem<Allocator>*)(a.allocate(size_alloc));
		if (p == NULL) {
			PVLOG_ERROR("(PVChunkMem): unable to allocate a new chunk !\n");
			return NULL;
		}
		new ((void*)p) PVChunkMem<Allocator>(a);
		p->_logical_end = p->begin();
		p->_physical_end = p->_logical_end + size;
		p->_source = parent;
		return p;
	}
	void free()
	{
		PVLOG_DEBUG("Deallocate chunk\n");
		alloc_chunk ap = _alloc;
		char* pbegin = begin();
		this->~PVChunkMem<Allocator>();
		ap.deallocate((char*)(this), sizeof(PVChunkMem)+(_physical_end-pbegin)+1);
	}
	PVChunk* realloc_grow(size_t n)
	{
		size_t cur_size = (size_t) ((uintptr_t)_physical_end - (uintptr_t)begin());
		size_t new_size = cur_size + n;
		PVChunkMem<Allocator>* ret = allocate(new_size, _source, _alloc);
		memcpy(ret->begin(), begin(), size());
		ret->_logical_end = ret->begin() + size();
		free();
		return ret;
	}

private:
	alloc_chunk _alloc;
};

}

#endif
