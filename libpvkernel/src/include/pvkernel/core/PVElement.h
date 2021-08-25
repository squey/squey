/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVELEMENT_FILE_H
#define PVELEMENT_FILE_H

#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVBufferSlice.h>
#include <pvkernel/core/PVField.h>

#include <list>

#include <tbb/scalable_allocator.h>

namespace PVCore
{

class PVTextChunk;

using list_fields =
    std::list<PVField, PVPreAllocatedListAllocator<PVField, tbb::scalable_allocator<PVField>>>;

class PVElement : public PVBufferSlice
{
	friend class PVField;
	friend class PVTextChunk;

  public:
	PVElement(PVTextChunk* parent, char* begin, char* end);
	explicit PVElement(PVTextChunk* parent);

  public:
	PVElement(PVElement const& src) = delete;

  public:
	~PVElement() override;

  public:
	bool valid() const;
	void set_invalid();
	bool filtered() const;
	void set_filtered();
	void set_parent(PVTextChunk* parent);
	PVTextChunk* chunk_parent();
	chunk_index get_elt_agg_index();
	size_t get_chunk_index() const { return _chunk_index; }

	list_fields& fields();
	list_fields const& c_fields() const;

	buf_list_t& realloc_bufs();

  public:
	// Element allocation and deallocation
	static inline PVElement* construct(PVTextChunk* parent, char* begin, char* end)
	{
		PVElement* ret = _alloc.allocate(1);
		new (ret) PVElement(parent, begin, end);
		return ret;
	}

	static inline PVElement* construct(PVTextChunk* parent)
	{
		PVElement* ret = _alloc.allocate(1);
		new (ret) PVElement(parent);
		return ret;
	}

	static inline void free(PVElement* elt)
	{
		_alloc.destroy(elt);
		_alloc.deallocate(elt, 1);
	}

  protected:
	// Set by the parent PVChunk
	void set_chunk_index(size_t i) { _chunk_index = i; }
	void init_fields(void* fields_buf, size_t size_buf);

  private:
	void init(PVTextChunk* parent);

  protected:
	bool _valid;
	bool _filtered;
	list_fields _fields;
	PVTextChunk* _parent;
	buf_list_t _reallocated_buffers; // buf_list_t defined in PVBufferSlice.h
	size_t _chunk_index;

  private:
	static tbb::scalable_allocator<PVElement> _alloc;
};
} // namespace PVCore

#endif
