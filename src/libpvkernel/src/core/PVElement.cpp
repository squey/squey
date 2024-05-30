//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVBufferSlice.h> // for buf_list_t, PVBufferSlice
#include <pvkernel/core/PVTextChunk.h>   // for PVChunk
#include <pvkernel/core/PVElement.h>     // for PVElement, list_fields

#include <pvbase/types.h> // for chunk_index

#include <cstddef> // for size_t
#include <new>     // for operator new
#include <utility> // for pair

#include <tbb/scalable_allocator.h> // for scalable_allocator
#include <tbb/tbb_allocator.h>      // for tbb_allocator

tbb::scalable_allocator<PVCore::PVElement> PVCore::PVElement::_alloc;

PVCore::PVElement::PVElement(PVTextChunk* parent) : PVBufferSlice(_reallocated_buffers)
{
	init(parent);
}

PVCore::PVElement::PVElement(PVTextChunk* parent, char* begin, char* end)
    : PVBufferSlice(begin, end, _reallocated_buffers)
{
	init(parent);
}

PVCore::PVElement::~PVElement()
{
	static tbb::tbb_allocator<char> alloc;
	for (auto v : _reallocated_buffers) {
		alloc.deallocate(v.first, v.second);
	}
}

void PVCore::PVElement::init(PVTextChunk* parent)
{
	_valid = true;
	_filtered = false;
	_parent = parent;
}

void PVCore::PVElement::init_fields(void* fields_buf, size_t size_buf)
{
	new (&_fields) list_fields(list_fields::allocator_type(fields_buf, size_buf));
	_fields.emplace_back(*this, begin(), end());
}

bool PVCore::PVElement::valid() const
{
	return _valid;
}

void PVCore::PVElement::set_invalid()
{
	_valid = false;
}

bool PVCore::PVElement::filtered() const
{
	return _filtered;
}

void PVCore::PVElement::set_filtered()
{
	_filtered = true;
}

PVCore::list_fields& PVCore::PVElement::fields()
{
	return _fields;
}

PVCore::list_fields const& PVCore::PVElement::c_fields() const
{
	return _fields;
}

void PVCore::PVElement::set_parent(PVTextChunk* parent)
{
	_parent = parent;
}

PVCore::PVTextChunk* PVCore::PVElement::chunk_parent()
{
	return _parent;
}

PVCore::buf_list_t& PVCore::PVElement::realloc_bufs()
{
	return _reallocated_buffers;
}

chunk_index PVCore::PVElement::get_elt_agg_index()
{
	PVTextChunk* parent = chunk_parent();
	return parent->get_agg_index_of_element(*this);
}
