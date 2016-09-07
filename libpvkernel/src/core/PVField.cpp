/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVBufferSlice.h> // for PVBufferSlice
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

PVCore::PVField::PVField(PVCore::PVElement& parent) : PVBufferSlice(parent.realloc_bufs())
{
	init(parent);
}

PVCore::PVField::PVField(PVCore::PVElement& parent, char* begin, char* end)
    : PVBufferSlice(begin, end, parent.realloc_bufs())
{
	init(parent);
}

void PVCore::PVField::init(PVElement& parent)
{
	_valid = true;
	_filtered = false;
	_parent = &parent;
}

bool PVCore::PVField::valid() const
{
	return _valid;
}

void PVCore::PVField::set_invalid()
{
	_valid = false;
}

bool PVCore::PVField::filtered() const
{
	return _filtered;
}

void PVCore::PVField::set_filtered()
{
	_filtered = true;
}

PVCore::PVElement* PVCore::PVField::elt_parent()
{
	return _parent;
}

size_t PVCore::PVField::get_agg_index_of_parent_element()
{
	return _parent->get_elt_agg_index();
}
