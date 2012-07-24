/**
 * \file PVSerializeArchiveOptions.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVSerializeArchiveOptions.h>
#include <pvkernel/core/PVFileSerialize.h>

#include <typeinfo>

PVCore::PVSerializeObject_p PVCore::PVSerializeArchiveOptions::create_object(QString const& name, PVSerializeObject* parent)
{
	PVSerializeObject_p ret(allocate_object(name, parent));
	return ret;
}


bool PVCore::PVSerializeArchiveOptions::must_write(PVSerializeObject const& parent, QString const& child)
{
	QString path = get_object_logical_path(parent);
	assert(_objects.contains(path));

	PVSerializeObject_p ours(_objects.value(path));
	PVSerializeObject_p our_child = ours->get_child_by_name(child);
	assert(our_child);
	assert(our_child->is_optional());
	return our_child->must_write();
}

void PVCore::PVSerializeArchiveOptions::include_all_files(bool inc)
{
	// AG: that's a little hack, since we should not be aware of what an original file is... !
	// No recursive function here. We look for PVFileSerialize objects, and include/exclude them.
	
	QList<PVSerializeObject_p> objs = _objects.values();
	foreach(PVSerializeObject_p o, objs) {
		if (o->bound_obj_type() == typeid(PVFileSerialize)) {
			o->set_write(inc);
		}
	}
}

int PVCore::PVSerializeArchiveOptions::does_include_all_files() const
{
	// AG: part of the previous hack
	
	int state = Qt::Unchecked;
	bool all_write = true;
	QList<PVSerializeObject_p> objs = _objects.values();
	foreach(PVSerializeObject_p o, objs) {
		if (o->bound_obj_type() == typeid(PVFileSerialize)) {
			if (o->must_write()) {
				state = Qt::PartiallyChecked;
			}
			else {
				all_write = false;
			}
		}
	}
	
	if (state == Qt::PartiallyChecked && all_write) {
		state = Qt::Checked;
	}

	return state;
}
