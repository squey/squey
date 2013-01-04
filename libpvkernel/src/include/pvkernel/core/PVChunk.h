/**
 * \file PVChunk.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCHUNK_FILE_H
#define PVCHUNK_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>
#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/rush/PVRawSourceBase_types.h>

#include <tbb/tbb_allocator.h>
#include <tbb/scalable_allocator.h>

#include <memory>
#include <cassert>
#include <list>

#include <sys/mman.h>

namespace PVRush {
	class PVAggregator;
	class PVNraw;
};

namespace PVCore {

typedef std::list< PVElement*, tbb::scalable_allocator<PVElement*> > list_elts;
//typedef std::list<PVElement*> list_elts;

// Describe chunk interface with no allocator template
// Useful in order to use chunks as function arguments...
/*! \brief Defines a chunk of data to be processed by the TBB pipeline during the normalisation process.
 *  
 * This class defines a chunk of data to be processed by the TBB pipeline during the normalisation process.
 *
 * A chunk is logically organised like this:
 * <ul>
 * <li>Chunk of data</li>
 * <li>
 *   <ul>
 *     <li>Element 1 (slice of chunk)
 *     <ul>
 *       <li>Field 1 (slice of element 1)</li>
 *       <li>Field 2 (slice of element 1)</li>
 *     </ul>
 *     </li>
 *     <li>Element 2 (slice of chunk) ...</li>
 *   </ul>
 * </li>
 * </ul>
 *
 * In memory, it is organised this way (see also \ref PVChunkMem): each PVChunkMem object is allocated in a way that it contains
 * space after its own structure for its data. So, a pointer to a PVChunkMem object actually points to:
 * <pre>
 * [PVChunkMem object] [space for data] 
 * ^
 * |
 * pointer to the PVChunkMem object
 * </pre>
 *
 * To have a direct pointer to the associated data of a PVChunk, use the \ref begin method. PVChunk is just an interface implemented by PVChunkMem.
 * PVChunkMem is a template object that takes an allocator as argument.
 *
 * \note
 * There is a concept of logical and physical end, that is the same as \ref PVBufferSlice. Refer to the documentation of this class for more information.
 *
 */
class LibKernelDecl PVChunk {
friend class PVRush::PVAggregator;

protected:
	// This is what a node in std::list<PVField> looks like !
	struct __node_list_field
	{
		PVCore::PVField f;
		void *p1;
		void *p2;
	};

public:
	PVChunk() : _index(0), _n_elts_invalid(0), _p_chunk_fields(NULL) {};
	virtual ~PVChunk()
	{
		free_structs();
	}
	void free_structs()
	{
		// Free elements
		list_elts::iterator it;
		for (it = _elts.begin(); it != _elts.end(); it++) {
			PVElement::free(*it);
		}
		free_fields_buffer();
		_elts.clear();
	}
public:
	/*! \brief Returns a pointer to the beggining of the associated data
	 */
	virtual char* begin() const = 0;

	/*! \brief Returns a pointer to the logical end of the associated data
	 */
	char* end() const { return _logical_end; };

	/*! \brief Returns a pointer to the physical end of the associated data
	 */
	char* physical_end() const { return _physical_end; };

	/*! \brief Set the logical end of the chunk.
	 *  \param[in] p Logical end of the chunk. Must be >= begin() and <= physical_end()
	 */
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
	inline void get_elts_stat(size_t& nelts_org, size_t& nelts_valid) const { nelts_org = _nelts_org; nelts_valid = _nelts_valid; }
	inline size_t get_nelts_valid() const { return _nelts_valid; }

	size_t size() const { return (size_t) ((uintptr_t)_logical_end - (uintptr_t)begin()); };
	size_t avail() const { return (size_t) ((uintptr_t)_physical_end - (uintptr_t)_logical_end); };

	virtual void free() = 0;
	virtual PVChunk* realloc_grow(size_t n) = 0;

	chunk_index get_index_of_element(PVElement const& elt) { return _get_idx_elt(elt) + _index; }
	chunk_index get_agg_index_of_element(PVElement const& elt) { return _get_idx_elt(elt) + _agg_index; }

	list_elts& elements() {return _elts;};
	list_elts const& c_elements() const {return _elts;}; 
	PVElement* add_element(char* start, char* end)
	{
		PVElement* new_elt = PVElement::construct(this, start, end);
		_elts.push_back(new_elt);
		return new_elt;
	}
	PVElement* add_element()
	{
		PVElement* new_elt = PVElement::construct(this);
		_elts.push_back(new_elt);
		return new_elt;
	}

	PVRush::PVRawSourceBase* source() const { return _source; };

	// Only visit one column
	template <typename F>
	void visit_column(const PVCol c, F const& f) const
	{
		PVRow r = 0;
		for (PVElement* elt: c_elements()) {
			if (!elt->valid()) {
				continue;
			}
			assert(c < elt->c_fields().size());
			PVCol cur_c = 0;
			for (PVField const& field: elt->fields()) {
				if (cur_c == c) {
					f(r, field);
					break;
				}
				cur_c++;
			}
			r++;
		}
	}

	// Column cache-aware visitor
	// TODO: at most eight field stream per line!
	template <typename F>
	void visit_by_column(F const& f) const
	{
		PVRow r = 0;
		for (PVElement* elt: c_elements()) {
			if (!elt->valid()) {
				continue;
			}
			PVCol c = 0;
			for (PVField const& field: elt->fields()) {
				f(r, c, field);
				c++;
			}
			r++;
		}
	}

	template <typename F>
	void visit_by_column(F const& f)
	{
		PVRow r = 0;
		for (PVElement* elt: elements()) {
			if (!elt->valid()) {
				continue;
			}
			PVCol c = 0;
			for (PVField& field: elt->fields()) {
				f(r, c, field);
				c++;
			}
			r++;
		}
	}

public:
	void set_elements_index()
	{
		size_t i = 0;
		list_elts::iterator it;
		for (it = _elts.begin(); it != _elts.end(); it++) {
			(*it)->set_chunk_index(i);
			i++;
		}
	}

	// This should be called when a chunk has been created to reserve its futur fields
	void init_elements_fields();

protected:
	void allocate_fields_buffer(PVRow nelts, PVCol nfields)
	{
		//_p_chunk_fields = ::malloc(sizeof(__node_list_field)*nfields*nelts);
		const size_t fields_size = sizeof(__node_list_field)*nfields*nelts;
		//_p_chunk_fields = mmap(NULL, _fields_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		_p_chunk_fields = tbb::scalable_allocator<char>().allocate(fields_size);
		//PVLOG_INFO("PVField %p allocate %u\n", _p_chunk_fields, sizeof(__node_list_field)*nfields*nelts);
	}

	void free_fields_buffer()
	{
		if (_p_chunk_fields) {
			//::free(_p_chunk_fields);
			//PVLOG_INFO("PVField %p deallocate %u\n", _p_chunk_fields, _fields_size);
			//munmap(_p_chunk_fields, _fields_size);
			tbb::scalable_allocator<char>().deallocate((char*) _p_chunk_fields, 0);
			_p_chunk_fields = NULL;
		}
	}

private:
	chunk_index _get_idx_elt(PVElement const& elt)
	{
		return elt.get_chunk_index();
	}

	PVCol get_source_number_fields() const;
protected:
	// Useful datas
	char* _logical_end;
	char* _physical_end;
	list_elts _elts;
	chunk_index _index;
	chunk_index _agg_index;
	PVRush::PVRawSourceBase *_source;
	PVRow _n_elts_invalid;
	size_t _nelts_org;
	size_t _nelts_valid;

	// Buffer containing the fields for this chunk
	void* _p_chunk_fields;
	//size_t _fields_size;
};


//template < template <class T> class Allocator = PVCore::PVMMapAllocator >
template < template <class T> class Allocator = tbb::scalable_allocator >
//template < template <class T> class Allocator = std::allocator >

class PVChunkMem : public PVChunk {
public:
	typedef Allocator<char> alloc_chunk;
private:
	PVChunkMem(alloc_chunk const& a) :
		PVChunk(), _alloc(a)
	{
	}
	virtual ~PVChunkMem() {}
public:
    char* begin() const { return (char*)(this+1); };
	static PVChunkMem* allocate(size_t size, PVRush::PVRawSourceBase* parent, alloc_chunk a = alloc_chunk())
	{
		size_t size_alloc = sizeof(PVChunkMem<Allocator>)+size+1;
		PVChunkMem<Allocator>* p = (PVChunkMem<Allocator>*)(a.allocate(size_alloc));
		//PVLOG_INFO("PVChunk alloc %u bytes, %p.\n", size_alloc, p);
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
		alloc_chunk ap = _alloc;
		char* pbegin = begin();
		this->~PVChunkMem<Allocator>();
		size_t dealloc = sizeof(PVChunkMem<Allocator>)+(_physical_end-pbegin)+1;
		//PVLOG_INFO("PVChunk dealloc %u bytes, %p.\n", dealloc, this);
		ap.deallocate((char*)(this), dealloc);
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
