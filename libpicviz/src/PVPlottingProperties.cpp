//! \file PVPlottingProperties.cpp
//! $Id: PVPlottingProperties.cpp 3062 2011-06-07 08:33:36Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVPlottingProperties.h>
#include <picviz/PVRoot.h>
#include <picviz/PVPlottingFilter.h>
#include <picviz/PVMapping.h>

#include <pvkernel/core/PVClassLibrary.h>

/******************************************************************************
 *
 * Picviz::PVPlottingProperties::PVPlottingProperties
 *
 *****************************************************************************/
Picviz::PVPlottingProperties::PVPlottingProperties(PVMapping const& mapping, PVRush::PVFormat const& format, PVCol idx):
	_mapping(&mapping)
{
	_index = idx;
	set_from_axis(format.get_axes().at(idx));
	_is_uptodate = false;
}

Picviz::PVPlottingProperties::PVPlottingProperties(PVMapping const& mapping, PVRush::PVAxisFormat const& axis, PVCol idx):
	_mapping(&mapping)
{
	_index = idx;
	set_from_axis(axis);
	_is_uptodate = false;
}

void Picviz::PVPlottingProperties::set_from_axis(PVRush::PVAxisFormat const& axis)
{
	QString mode = axis.get_plotting();
	set_args(axis.get_args_plotting());
	set_mode(mode);
}

void Picviz::PVPlottingProperties::set_mapping(const PVMapping& mapping)
{
	_mapping = &mapping;
	_type = get_type();
	_is_uptodate = false;
	set_mode(_mode);
}

QString Picviz::PVPlottingProperties::get_type() const
{
	assert(_mapping);
	return _mapping->get_type_for_col(_index);
}

void Picviz::PVPlottingProperties::set_args(PVCore::PVArgumentList const& args)
{
	if (!PVCore::comp_hash(_args, args)) {
		_is_uptodate = false;
	}
	_args = args;
	if (_plotting_filter) {
		_plotting_filter->set_args(args);
	}
}

void Picviz::PVPlottingProperties::set_mode(QString const& mode)
{
	if (_is_uptodate) {
		_is_uptodate = (_mode == mode);
	}
	_mode = mode;
	PVPlottingFilter::p_type lib_filter = LIB_CLASS(PVPlottingFilter)::get().get_class_by_name(get_type() + "_" + mode);
	if (!lib_filter) {
		PVLOG_WARN("Plotting mode '%s' for type '%s' does not exist ! Falling back to default mode...\n", qPrintable(mode), qPrintable(get_type()));
		_mode = "default";
		_is_uptodate = false;
		lib_filter = LIB_CLASS(PVPlottingFilter)::get().get_class_by_name(get_type() + "_" + _mode);
		if (!lib_filter) {
			PVLOG_ERROR("Plotting mode '%s' for type '%s' does not exist !\n", qPrintable(mode), qPrintable(get_type()));
		}
		_plotting_filter = lib_filter->clone<PVPlottingFilter>();
	}
	else {
		_plotting_filter = lib_filter->clone<PVPlottingFilter>();
		_plotting_filter->set_args(_args);
	}
	_type = get_type();
}

Picviz::PVPlottingFilter::p_type Picviz::PVPlottingProperties::get_plotting_filter()
{
	// If type has changed, reload the good plugin
	if (_type != get_type()) {
		set_mode(_mode);
		_is_uptodate = false;
	}
	return _plotting_filter;
}

bool Picviz::PVPlottingProperties::operator==(PVPlottingProperties const& org)
{
	return (_plotting_filter == org._plotting_filter) && (_index == org._index);
}

void Picviz::PVPlottingProperties::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("index", _index);
	so.attribute("mode", _mode);
	so.arguments("properties", _args);

	if (!so.is_writing()) {
		_is_uptodate = false;
	}
}
