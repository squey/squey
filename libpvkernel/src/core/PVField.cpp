#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVElementData.h>

PVCore::PVField::PVField(PVCore::PVElement const& parent, char* begin, char* end) :
	PVBufferSlice(begin, end, parent.realloc_bufs())
{
	_valid = true;
	_parent = parent.d.data();
}

bool PVCore::PVField::valid() const
{
	return _valid;
}

void PVCore::PVField::set_invalid()
{
	_valid = false;
}

PVCore::PVElement* PVCore::PVField::elt_parent()
{
	return _parent->_elt;
}

void PVCore::PVField::set_parent(PVCore::PVElement const& parent)
{
	_parent = parent.d.data();
}

void PVCore::PVField::deep_copy()
{
	_realloc_data();
}

size_t PVCore::PVField::get_index_of_parent_element()
{
	return _parent->_elt->get_elt_index();
}

size_t PVCore::PVField::get_agg_index_of_parent_element()
{
	return _parent->_elt->get_elt_agg_index();
}
