/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMappingFilter.h>
#include <pvkernel/rush/PVFormat.h>

QStringList Inendi::PVMappingFilter::list_modes(std::string const& type)
{
	LIB_CLASS(Inendi::PVMappingFilter)
	::list_classes const& map_filters = LIB_CLASS(Inendi::PVMappingFilter)::get().get_list();
	QStringList ret;
	for (auto it = map_filters.begin(); it != map_filters.end(); it++) {
		auto const& filter = it->value();
		auto usable_types = filter->list_usable_type();
		if (usable_types.find(type) != usable_types.end()) {
			ret << it->key();
		}
	}
	return ret;
}
