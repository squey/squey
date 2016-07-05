/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVPlottingProperties.h>
#include <inendi/PVPlottingFilter.h>

#include <pvkernel/core/PVClassLibrary.h>

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
{
	_index = idx;
	Inendi::PVAxis axis(axis_format);

	QString mode = axis.get_plotting();
	set_args(axis.get_args_plotting());
	set_mode(mode);
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

void Inendi::PVPlottingProperties::set_mode(QString const& mode)
{
	if (_is_uptodate) {
		_is_uptodate = (_mode == mode);
	}

	_mode = mode;
	PVPlottingFilter::p_type lib_filter =
	    LIB_CLASS(PVPlottingFilter)::get().get_class_by_name(mode);

	_plotting_filter = lib_filter->clone<PVPlottingFilter>();
	set_args(_args);
}

Inendi::PVPlottingFilter::p_type Inendi::PVPlottingProperties::get_plotting_filter()
{
	return _plotting_filter;
}

bool Inendi::PVPlottingProperties::operator==(PVPlottingProperties const& org)
{
	return (_plotting_filter == org._plotting_filter) && (_index == org._index);
}

void Inendi::PVPlottingProperties::serialize(PVCore::PVSerializeObject& so,
                                             PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("index", _index);
	so.attribute("mode", _mode);

	if (!so.is_writing()) {
		_is_uptodate = false;
	}
}
