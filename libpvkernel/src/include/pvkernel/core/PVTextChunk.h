/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVTextChunk_FILE_H
#define PVTextChunk_FILE_H

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>             // for PVElement
#include <pvkernel/core/PVField.h>               // for PVField
#include <pvkernel/core/PVLogger.h>              // for PVLOG_ERROR
#include <pvkernel/rush/PVRawSourceBase_types.h> // for EChunkType

#include <tbb/scalable_allocator.h> // for scalable_allocator

#include <cassert> // for assert
#include <cstddef> // for size_t
#include <cstdint> // for uintptr_t
#include <cstring> // for memcpy
#include <list>    // for _List_iterator, etc
#include <new>     // for operator new

namespace PVRush
{
class PVAggregator;
class PVRawSourceBase;
} // namespace PVRush
;

namespace PVCore
{

typedef std::list<PVElement*, tbb::scalable_allocator<PVElement*>> list_elts;
// typedef std::list<PVElement*> list_elts;

// Describe chunk interface with no allocator template
// Useful in order to use chunks as function arguments...
/*! \brief Defines a chunk of data to be processed by the TBB pipeline during the normalisation
 *process.
 *
 * This class defines a chunk of data to be processed by the TBB pipeline during the normalisation
 *process.
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
 * In memory, it is organised this way (see also \ref PVTextChunkMem): each PVTextChunkMem object is
 *allocated in a way that it contains
 * space after its own structure for its data. So, a pointer to a PVTextChunkMem object actually
 *points
 *to:
 * <pre>
 * [PVTextChunkMem object] [space for data]
 * ^
 * |
 * pointer to the PVTextChunkMem object
 * </pre>
 *
 * To have a direct pointer to the associated data of a PVTextChunk, use the \ref begin method.
 *PVTextChunk
 *is just an interface implemented by PVTextChunkMem.
 * PVTextChunkMem is a template object that takes an allocator as argument.
 *
 * \note
 * There is a concept of logical and physical end, that is the same as \ref PVBufferSlice. Refer to
 *the documentation of this class for more information.
 *
 */
class PVTextChunk : public PVChunk
{
	friend class PVRush::PVAggregator;

  protected:
	// This is what a node in std::list<PVField> looks like !
	struct __node_list_field {
		PVCore::PVField f;
		void* p1;
		void* p2;
	};

  public:
	PVTextChunk() : _p_chunk_fields(nullptr){};
	virtual ~PVTextChunk() { free_structs(); }

	static constexpr PVRush::EChunkType chunk_type = PVRush::EChunkType::TEXT;

  public:
	size_t rows_count() const override { return c_elements().size(); };

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
	inline void set_agg_index(chunk_index v) { _agg_index = v; }
	inline void set_index(chunk_index i) { _index = i; };
	inline void set_elts_stat(size_t nelts_org, size_t nelts_valid)
	{
		_nelts_org = nelts_org;
		_nelts_valid = nelts_valid;
	}
	inline void get_elts_stat(size_t& nelts_org, size_t& nelts_valid) const
	{
		nelts_org = _nelts_org;
		nelts_valid = _nelts_valid;
	}
	inline size_t get_nelts_valid() const { return _nelts_valid; }

	size_t size() const { return (size_t)((uintptr_t)_logical_end - (uintptr_t)begin()); };
	size_t avail() const { return (size_t)((uintptr_t)_physical_end - (uintptr_t)_logical_end); };

	virtual PVTextChunk* realloc_grow(size_t n) = 0;

	chunk_index get_agg_index_of_element(PVElement const& elt)
	{
		return elt.get_chunk_index() + _agg_index;
	}

	list_elts& elements() { return _elts; };
	list_elts const& c_elements() const { return _elts; };
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

	void remove_nelts_front(size_t nelts_remove, size_t nstart_rem = 0) override
	{
		auto it_elt = elements().begin();
		// Go to the nstart_rem ith element
		for (chunk_index i = 0; i < nstart_rem; i++) {
			it_elt++;
		}
		for (chunk_index i = 0; i < nelts_remove; i++) {
			PVCore::PVElement::free(*it_elt);
			auto it_er = it_elt;
			it_elt++;
			elements().erase(it_er);
		}
	}

	PVRush::PVRawSourceBase* source() const { return _source; };

  public:
	void set_elements_index() override
	{
		size_t i = 0;
		for (auto& elt : _elts) {
			elt->set_chunk_index(i++);
		}
	}

	// This should be called when a chunk has been created to reserve its futur fields
	void init_elements_fields();

  protected:
	void allocate_fields_buffer(PVRow nelts, PVCol nfields)
	{
		//_p_chunk_fields = ::malloc(sizeof(__node_list_field)*nfields*nelts);
		const size_t fields_size = sizeof(__node_list_field) * nfields.value() * nelts;
		//_p_chunk_fields = mmap(nullptr, _fields_size, PROT_READ | PROT_WRITE, MAP_PRIVATE |
		// MAP_ANONYMOUS, -1, 0);
		_p_chunk_fields = tbb::scalable_allocator<char>().allocate(fields_size);
		// PVLOG_INFO("PVField %p allocate %u\n", _p_chunk_fields,
		// sizeof(__node_list_field)*nfields*nelts);
	}

	void free_fields_buffer()
	{
		if (_p_chunk_fields) {
			//::free(_p_chunk_fields);
			// PVLOG_INFO("PVField %p deallocate %u\n", _p_chunk_fields, _fields_size);
			// munmap(_p_chunk_fields, _fields_size);
			tbb::scalable_allocator<char>().deallocate((char*)_p_chunk_fields, 0);
			_p_chunk_fields = nullptr;
		}
	}

  protected:
	// Useful datas
	char* _logical_end;
	char* _physical_end;
	list_elts _elts;
	PVRush::PVRawSourceBase* _source;
	size_t _nelts_org;
	size_t _nelts_valid;

	// Buffer containing the fields for this chunk
	void* _p_chunk_fields;
	// size_t _fields_size;
};

// template < template <class T> class Allocator = PVCore::PVMMapAllocator >
template <template <class T> class Allocator = tbb::scalable_allocator>
// template < template <class T> class Allocator = std::allocator >

class PVTextChunkMem : public PVTextChunk
{
  public:
	typedef Allocator<char> alloc_chunk;
	static_assert(sizeof(alloc_chunk) == 1, "Bad begin accessor");

  private:
	explicit PVTextChunkMem(alloc_chunk const& a) : PVTextChunk(), _alloc(a) {}
	~PVTextChunkMem() override = default;

  public:
	char* begin() const override { return (char*)(this + 1); };
	static PVTextChunkMem*
	allocate(size_t size, PVRush::PVRawSourceBase* parent, alloc_chunk a = alloc_chunk())
	{
		size_t size_alloc = sizeof(PVTextChunkMem<Allocator>) + size + 1;
		PVTextChunkMem<Allocator>* p = (PVTextChunkMem<Allocator>*)(a.allocate(size_alloc));
		// PVLOG_INFO("PVTextChunk alloc %u bytes, %p.\n", size_alloc, p);
		if (p == nullptr) {
			PVLOG_ERROR("(PVTextChunkMem): unable to allocate a new chunk !\n");
			return nullptr;
		}
		new ((void*)p) PVTextChunkMem<Allocator>(a);
		p->_logical_end = p->begin();
		p->_physical_end = p->_logical_end + size;
		p->_source = parent;
		return p;
	}
	void free() override
	{
		alloc_chunk ap = _alloc;
		char* pbegin = begin();
		this->~PVTextChunkMem<Allocator>();
		size_t dealloc = sizeof(PVTextChunkMem<Allocator>) + (_physical_end - pbegin) + 1;
		// PVLOG_INFO("PVTextChunk dealloc %u bytes, %p.\n", dealloc, this);
		ap.deallocate((char*)(this), dealloc);
	}
	PVTextChunk* realloc_grow(size_t n) override
	{
		size_t cur_size = (size_t)((uintptr_t)_physical_end - (uintptr_t)begin());
		size_t new_size = cur_size + n;
		PVTextChunkMem<Allocator>* ret = allocate(new_size, _source, _alloc);
		memcpy(ret->begin(), begin(), cur_size);
		ret->_logical_end = ret->begin() + cur_size;
		ret->_index = _index;
		ret->_agg_index = _agg_index;
		ret->_init_size = _init_size;
		free();
		return ret;
	}

  private:
	alloc_chunk _alloc;
};
} // namespace PVCore

#endif
