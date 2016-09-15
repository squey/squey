/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVAxis.h>               // for PVAxis
#include <inendi/PVPlottingFilter.h>     // for PVPlottingFilter, etc
#include <inendi/PVPlottingProperties.h> // for PVPlottingProperties

#include <pvkernel/rush/PVAxisFormat.h> // for PVAxisFormat
#include <pvkernel/rush/PVFormat.h>     // for PVFormat

#include <pvkernel/core/PVArgument.h>     // for PVArgumentList, etc
#include <pvkernel/core/PVClassLibrary.h> // for LIB_CLASS, etc
#include <pvkernel/core/PVCompList.h>     // for comp_hash
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/core/PVSerializeObject.h> // for PVSerializeObject

#include "pvbase/types.h" // for PVCol

#include <QList>   // for QList
#include <QString> // for QString

#include <memory> // for __shared_ptr
#include <string> // for string, operator==, etc

/******************************************************************************
 *
 * Inendi::PVPlottingProperties::PVPlottingProperties
 *
 *****************************************************************************/
Inendi::PVPlottingProperties::PVPlottingProperties(PVRush::PVFormat const& format, PVCol idx)
    : PVPlottingProperties(format.get_axes().at(idx))
{
}

Inendi::PVPlottingProperties::PVPlottingProperties(PVRush::PVAxisFormat const& axis_format)
    : PVPlottingProperties(Inendi::PVAxis(axis_format).get_plotting().toStdString(),
                           Inendi::PVAxis(axis_format).get_args_plotting())
{
}

Inendi::PVPlottingProperties::PVPlottingProperties(std::string const& mode,
                                                   PVCore::PVArgumentList args)
    : _mode(mode)
    , _plotting_filter(LIB_CLASS(Inendi::PVPlottingFilter)::get()
                           .get_class_by_name(QString::fromStdString(_mode))
                           ->clone<PVPlottingFilter>())
    , _is_uptodate(false)
{
	set_args(args);
}

void Inendi::PVPlottingProperties::set_args(PVCore::PVArgumentList const& args)
{
	if (!PVCore::comp_hash(_args, args)) {
		_is_uptodate = false;
	}
	if (_plotting_filter) {
		PVCore::PVArgumentList new_args = _plotting_filter->get_default_args();
		PVArgumentList_set_common_args_from(new_args, args);
		_plotting_filter->set_args(new_args);
		_args = new_args;
	} else {
		_args = args;
	}
}

void Inendi::PVPlottingProperties::set_mode(std::string const& mode)
{
	if (_is_uptodate) {
		_is_uptodate = (_mode == mode);
	}

	_mode = mode;
	PVPlottingFilter::p_type lib_filter =
	    LIB_CLASS(PVPlottingFilter)::get().get_class_by_name(QString::fromStdString(mode));

	_plotting_filter = lib_filter->clone<PVPlottingFilter>();
	set_args(_args);
}

Inendi::PVPlottingFilter::p_type Inendi::PVPlottingProperties::get_plotting_filter()
{
	return _plotting_filter;
}

Inendi::PVPlottingProperties
Inendi::PVPlottingProperties::serialize_read(PVCore::PVSerializeObject& so)
{
	QString mode;
	so.attribute("mode", mode);

	PVCore::PVArgumentList args;
	so.arguments("properties", args, args);
	return {mode.toStdString(), args};
}

void Inendi::PVPlottingProperties::serialize_write(PVCore::PVSerializeObject& so)
{
	QString mode = QString::fromStdString(_mode);
	so.attribute("mode", mode);
	so.arguments("properties", _args, _plotting_filter->get_default_args());
}
