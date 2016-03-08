/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMappingFilter.h>
#include <pvkernel/rush/PVFormat.h>

Inendi::PVMappingFilter::PVMappingFilter()
{
	_dest = NULL;
}

void Inendi::PVMappingFilter::init()
{
}

void Inendi::PVMappingFilter::set_dest_array(PVRow size, decimal_storage_type* ptr)
{
	assert(ptr);
	// This array is supposed to be as large as the values given to operator()
	_dest = ptr;
	_dest_size = size;
}

QStringList Inendi::PVMappingFilter::list_types()
{
	LIB_CLASS(PVMappingFilter)::list_classes const& map_filters = LIB_CLASS(PVMappingFilter)::get().get_list();
	LIB_CLASS(PVMappingFilter)::list_classes::const_iterator it;
	QStringList ret;
	for (it = map_filters.begin(); it != map_filters.end(); it++) {
		QString const& name = it->key();
		QStringList params = name.split('_');
		if (!ret.contains(params[0])) {
			ret << params[0];
		}
	}
    return ret;
}

QStringList Inendi::PVMappingFilter::list_modes(QString const& type)
{
	LIB_CLASS(Inendi::PVMappingFilter)::list_classes const& map_filters = LIB_CLASS(Inendi::PVMappingFilter)::get().get_list();
	LIB_CLASS(Inendi::PVMappingFilter)::list_classes::const_iterator it;
	QStringList ret;
	for (it = map_filters.begin(); it != map_filters.end(); it++) {
		QString const& name = it->key();
		QStringList params = name.split('_');
		if (params[0].compare(type) == 0) {
			ret << params[1];
		}
	}
    return ret;
}

QString Inendi::PVMappingFilter::get_human_name() const
{
	QStringList params = registered_name().split('_');
	return params[1];
}
