/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVPlottingProperties.h>
#include <inendi/PVPlottingFilter.h>
#include <inendi/PVMapping.h>

#include <pvkernel/core/PVClassLibrary.h>

/******************************************************************************
 *
 * Inendi::PVPlottingProperties::PVPlottingProperties
 *
 *****************************************************************************/
Inendi::PVPlottingProperties::PVPlottingProperties(PVMapping const& mapping,
                                                   PVRush::PVFormat const& format,
                                                   PVCol idx)
    : _mapping(&mapping)
{
	_index = idx;
	set_from_axis(format.get_axes().at(idx));
}

Inendi::PVPlottingProperties::PVPlottingProperties(PVMapping const& mapping,
                                                   PVRush::PVAxisFormat const& axis,
                                                   PVCol idx)
    : _mapping(&mapping)
{
	_index = idx;
	set_from_axis(axis);
}

void Inendi::PVPlottingProperties::set_from_axis(PVRush::PVAxisFormat const& axis)
{
	set_from_axis(Inendi::PVAxis(axis));
}

void Inendi::PVPlottingProperties::set_from_axis(Inendi::PVAxis const& axis)
{
	QString mode = axis.get_plotting();
	set_args(axis.get_args_plotting());
	set_mode(mode);
}

void Inendi::PVPlottingProperties::set_mapping(const PVMapping& mapping)
{
	_mapping = &mapping;
	_type = get_type();
	_is_uptodate = false;
	set_mode(_mode);
}

QString Inendi::PVPlottingProperties::get_type() const
{
	assert(_mapping);
	return _mapping->get_type_for_col(_index);
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
	    LIB_CLASS(PVPlottingFilter)::get().get_class_by_name(get_type() + "_" + mode);

	_plotting_filter = lib_filter->clone<PVPlottingFilter>();
	set_args(_args);

	_type = get_type();
}

Inendi::PVPlottingFilter::p_type Inendi::PVPlottingProperties::get_plotting_filter()
{
	// If type has changed, reload the good plugin
	if (_type != get_type()) {
		set_mode(_mode);
		_is_uptodate = false;
	}
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
	/*if (_plotting_filter) {
	        so.arguments("properties", _args, _plotting_filter->default_args());
	}*/
}
