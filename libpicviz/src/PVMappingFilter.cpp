/**
 * \file PVMappingFilter.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <picviz/PVMappingFilter.h>
#include <pvkernel/rush/PVFormat.h>

#include <pvkernel/core/stdint.h>

Picviz::PVMappingFilter::PVMappingFilter()
{
	_dest = NULL;
	_grp_value = NULL;
}

void Picviz::PVMappingFilter::init()
{
}

void Picviz::PVMappingFilter::set_dest_array(PVRow size, decimal_storage_type* ptr)
{
	assert(ptr);
	// This array is supposed to be as large as the values given to operator()
	_dest = ptr;
	_dest_size = size;
}

QStringList Picviz::PVMappingFilter::list_types()
{
	LIB_CLASS(PVMappingFilter)::list_classes const& map_filters = LIB_CLASS(PVMappingFilter)::get().get_list();
	LIB_CLASS(PVMappingFilter)::list_classes::const_iterator it;
	QStringList ret;
	for (it = map_filters.begin(); it != map_filters.end(); it++) {
		QString const& name = it.key();
		QStringList params = name.split('_');
		if (!ret.contains(params[0])) {
			ret << params[0];
		}
	}
    return ret;
}

QStringList Picviz::PVMappingFilter::list_modes(QString const& type)
{
	LIB_CLASS(Picviz::PVMappingFilter)::list_classes const& map_filters = LIB_CLASS(Picviz::PVMappingFilter)::get().get_list();
	LIB_CLASS(Picviz::PVMappingFilter)::list_classes::const_iterator it;
	QStringList ret;
	for (it = map_filters.begin(); it != map_filters.end(); it++) {
		QString const& name = it.key();
		QStringList params = name.split('_');
		if (params[0].compare(type) == 0) {
			ret << params[1];
		}
	}
    return ret;
}

QString Picviz::PVMappingFilter::get_human_name() const
{
	QStringList params = registered_name().split('_');
	return params[1];
}
