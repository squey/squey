/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

tbb::scalable_allocator<PVCore::PVElement> PVCore::PVElement::_alloc;
// std::allocator<PVCore::PVElement> PVCore::PVElement::_alloc;

PVCore::PVElement::PVElement(PVChunk* parent) : PVBufferSlice(_reallocated_buffers)
{
	init(parent);
}

PVCore::PVElement::PVElement(PVChunk* parent, char* begin, char* end)
    : PVBufferSlice(begin, end, _reallocated_buffers)
{
	init(parent);
}

PVCore::PVElement::PVElement(PVElement const& src) : PVBufferSlice(src)
{
	// No copy must occur !
	assert(false);
}

PVCore::PVElement::~PVElement()
{
	static tbb::tbb_allocator<char> alloc;
	buf_list_t::const_iterator it;
	for (it = _reallocated_buffers.begin(); it != _reallocated_buffers.end(); it++) {
		alloc.deallocate(it->first, it->second);
	}
}

void PVCore::PVElement::init(PVChunk* parent)
{
	_valid = true;
	_filtered = false;
	_parent = parent;
}

void PVCore::PVElement::init_fields(void* fields_buf, size_t size_buf)
{
	new (&_fields) list_fields(list_fields::allocator_type(fields_buf, size_buf));
	_fields.push_back(PVField(*this, begin(), end()));
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

void PVCore::PVElement::set_parent(PVChunk* parent)
{
	_parent = parent;
}

PVCore::PVChunk* PVCore::PVElement::chunk_parent()
{
	return _parent;
}

PVCore::buf_list_t& PVCore::PVElement::realloc_bufs()
{
	return _reallocated_buffers;
}

chunk_index PVCore::PVElement::get_elt_agg_index()
{
	PVChunk* parent = chunk_parent();
	return parent->get_agg_index_of_element(*this);
}
