/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVLogger.h>

#include <inendi/PVPlottingFilter.h>

Inendi::PVPlottingFilter::PVPlottingFilter()
    : PVFilter::PVFilterFunctionBase<uint32_t*, pvcop::db::array const&>()
    , PVCore::PVRegistrableClass<Inendi::PVPlottingFilter>()
{
	_dest = NULL;
	_dest_size = 0;
}

void Inendi::PVPlottingFilter::set_dest_array(PVRow size, uint32_t* array)
{
	assert(array);
	_dest_size = size;
	_dest = array;
}

QString Inendi::PVPlottingFilter::mode_from_registered_name(QString const& rn)
{
	QStringList params = rn.split('_');
	return params[1];
}

QString Inendi::PVPlottingFilter::get_human_name() const
{
	return mode_from_registered_name(registered_name());
}
