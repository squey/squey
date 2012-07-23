/**
 * \file PVField.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVElement.h>

PVCore::PVField::PVField(PVCore::PVElement& parent):
	PVBufferSlice(parent.realloc_bufs())
{
	init(parent);
}

PVCore::PVField::PVField(PVCore::PVElement& parent, char* begin, char* end):
	PVBufferSlice(begin, end, parent.realloc_bufs())
{
	init(parent);
}

void PVCore::PVField::init(PVElement& parent)
{
	_valid = true;
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

PVCore::PVElement* PVCore::PVField::elt_parent()
{
	return _parent;
}

void PVCore::PVField::set_parent(PVCore::PVElement& parent)
{
	_parent = &parent;
	set_buflist(parent.realloc_bufs());
}

void PVCore::PVField::deep_copy()
{
	_realloc_data();
}

size_t PVCore::PVField::get_index_of_parent_element()
{
	return _parent->get_elt_index();
}

size_t PVCore::PVField::get_agg_index_of_parent_element()
{
	return _parent->get_elt_agg_index();
}
