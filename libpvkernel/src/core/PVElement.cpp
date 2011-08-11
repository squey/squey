#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVElementData.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVChunk.h>

PVCore::PVElement::PVElement(PVChunk* parent) :
	d(new PVElementData()),
	PVBufferSlice(d->_reallocated_buffers)
{
	init(parent);
}

PVCore::PVElement::PVElement(PVChunk* parent, char* begin, char* end) :
	d(new PVElementData()),
	PVBufferSlice(begin, end, d->_reallocated_buffers)
{
	init(parent);
}

PVCore::PVElement::PVElement(PVElement const& src) :
	PVBufferSlice(src),
	d(src.d)
{
	d->_elt = this;
}

PVCore::PVElement::~PVElement()
{
	clear_saved_buf();
}

void PVCore::PVElement::init(PVChunk* parent)
{
	d->_valid = true;
	d->_parent = parent;
	d->_elt = this;
	// In the beggining, it only has a big field
	PVField f(*this, begin(), end());
	d->_fields.push_back(f);
	d->_org_buf = NULL;
	d->_org_buf_size = 0;
}

PVCore::PVElement& PVCore::PVElement::operator=(PVElement const& src)
{
	d = src.d;
	d->_elt = this;
	return *this;
}

bool PVCore::PVElement::valid() const
{
	return d->_valid;
}

void PVCore::PVElement::set_invalid()
{
	d->_valid = false;
}

PVCore::list_fields& PVCore::PVElement::fields()
{
	return d->_fields;
}

PVCore::list_fields const& PVCore::PVElement::c_fields() const
{
	return d->_fields;
}

void PVCore::PVElement::set_parent(PVChunk* parent)
{
	d->_parent = parent;
}

PVCore::PVChunk* PVCore::PVElement::chunk_parent()
{
	return d->_parent;
}

PVCore::buf_list_t& PVCore::PVElement::realloc_bufs() const
{
	return d->_reallocated_buffers;
}

void PVCore::PVElement::save_elt_buffer()
{
	clear_saved_buf();
	static tbb::scalable_allocator<char> alloc;
	d->_org_buf = alloc.allocate(size());
	d->_org_buf_size = size();
	memcpy(d->_org_buf, begin(), size());
}

void PVCore::PVElement::clear_saved_buf()
{
	if (!d->_org_buf) {
		return;
	}

	static tbb::scalable_allocator<char> alloc;
	alloc.deallocate(d->_org_buf, d->_org_buf_size);
	d->_org_buf = NULL;
	d->_org_buf_size = 0;
}

char* PVCore::PVElement::get_saved_elt_buffer(size_t& n)
{
	n = d->_org_buf_size;
	return d->_org_buf;
}

chunk_index PVCore::PVElement::get_elt_index()
{
	PVChunk* parent = chunk_parent();
	return parent->get_index_of_element(*this);
}

chunk_index PVCore::PVElement::get_elt_agg_index()
{
	PVChunk* parent = chunk_parent();
	return parent->get_agg_index_of_element(*this);
}
