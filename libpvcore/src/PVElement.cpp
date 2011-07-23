#include <pvcore/PVElement.h>
#include <pvcore/PVElementData.h>
#include <pvcore/PVField.h>
#include <pvcore/PVChunk.h>

PVCore::PVElement::PVElement(PVChunk* parent, char* begin, char* end) :
	d(new PVElementData()),
	PVBufferSlice(begin, end, d->_reallocated_buffers)
{
	d->_valid = true;
	d->_parent = parent;
	d->_elt = this;
	// In the beggining, it only has a big field
	PVField f(*this, begin, end);
	d->_fields.push_back(f);
}

PVCore::PVElement::PVElement(PVElement const& src) :
	PVBufferSlice(src),
	d(src.d)
{
	d->_elt = this;
}

PVCore::PVElement::~PVElement()
{
}

void PVCore::PVElement::deep_copy()
{
	_realloc_data();
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
