/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVPlottingFilter.h>
#include <inendi/PVPlottingProperties.h>

#include <pvkernel/core/PVClassLibrary.h>

#include <pvkernel/rush/PVFormatVersion.h>

/******************************************************************************
 *
 * Inendi::PVPlottingProperties::PVPlottingProperties
 *
 *****************************************************************************/
Inendi::PVPlottingProperties::PVPlottingProperties(PVRush::PVFormat const& format, PVCol idx)
    : PVPlottingProperties(format.get_axes().at(idx), idx)
{
}

Inendi::PVPlottingProperties::PVPlottingProperties(PVRush::PVAxisFormat const& axis_format,
                                                   PVCol idx)
    : PVPlottingProperties(Inendi::PVAxis(axis_format).get_plotting().toStdString(),
                           Inendi::PVAxis(axis_format).get_args_plotting(),
                           idx)
{
}

Inendi::PVPlottingProperties::PVPlottingProperties(std::string const& mode,
                                                   PVCore::PVArgumentList args,
                                                   PVCol idx)
    : _mode(mode)
    , _index(idx)
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

bool Inendi::PVPlottingProperties::operator==(PVPlottingProperties const& org) const
{
	return (_plotting_filter == org._plotting_filter) && (_index == org._index);
}

Inendi::PVPlottingProperties
Inendi::PVPlottingProperties::serialize_read(PVCore::PVSerializeObject& so)
{
	PVCol idx;
	so.attribute("index", idx);

	QString mode;
	so.attribute("mode", mode);

	PVCore::PVArgumentList args;
	so.arguments("properties", args, args);
	return {mode.toStdString(), args, idx};
}

void Inendi::PVPlottingProperties::serialize_write(PVCore::PVSerializeObject& so)
{
	so.attribute("index", _index);
	QString mode = QString::fromStdString(_mode);
	so.attribute("mode", mode);
	so.arguments("properties", _args, _plotting_filter->get_default_args());
}
