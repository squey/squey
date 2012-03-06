//! \file PVMappingProperties.cpp
//! $Id: PVMappingProperties.cpp 3062 2011-06-07 08:33:36Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVMapping.h>
#include <picviz/PVMappingProperties.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <picviz/PVRoot.h>

Picviz::PVMappingProperties::PVMappingProperties(PVRush::PVFormat const& format, PVCol idx)
{
	_index = idx;
	set_from_axis(format.get_axes().at(idx));
}

Picviz::PVMappingProperties::PVMappingProperties(PVRush::PVAxisFormat const& axis, PVCol idx)
{
	_index = idx;
	set_from_axis(axis);
}

void Picviz::PVMappingProperties::set_from_axis(PVRush::PVAxisFormat const& axis)
{
	QString type = axis.get_type();
	QString mode = axis.get_mapping();
	QString group = axis.get_group();
	PVCore::PVArgumentList args = axis.get_args_mapping();

	set_args(args);
	set_type(type, mode);

	if (!group.isEmpty()) {
		_group_key = group + "_" + _type;
	}
}

void Picviz::PVMappingProperties::set_type(QString const& type, QString const& mode)
{
	if (_is_uptodate && _type == type && _mode == mode) {
		return;
	}
	_type = type;
	_is_uptodate = false;
	set_mode(mode);
}

void Picviz::PVMappingProperties::set_args(PVCore::PVArgumentList const& args)
{
	if (!PVCore::comp_hash(args, _args)) {
		_is_uptodate = false;
	}
	if (_mapping_filter) {
		PVCore::PVArgumentList new_args = _mapping_filter->get_default_args();
		PVArgumentList_set_common_args_from(new_args, args);
		_mapping_filter->set_args(new_args);
		_args = new_args;
	}
	else {
		_args = args;
	}
}

void Picviz::PVMappingProperties::set_mode(QString const& mode)
{
	if (_is_uptodate && _mode == mode) {
		return;
	}
	_is_uptodate = false;
	_mode = mode;
	PVMappingFilter::p_type lib_filter = LIB_CLASS(Picviz::PVMappingFilter)::get().get_class_by_name(_type + "_" + mode);
	if (!lib_filter) {
		PVLOG_WARN("Mapping '%s' for type '%s' does not exist ! Falling back to default mode...\n", qPrintable(mode), qPrintable(_type));
		_mode = "default";
		_is_uptodate = false;
		lib_filter = LIB_CLASS(PVMappingFilter)::get().get_class_by_name(get_type() + "_" + _mode);
		if (!lib_filter) {
			PVLOG_ERROR("Mapping mode '%s' for type '%s' does not exist !\n", qPrintable(mode), qPrintable(get_type()));
		}
		_mapping_filter = lib_filter->clone<PVMappingFilter>();
	}
	else {
		_mapping_filter = lib_filter->clone<PVMappingFilter>();
		set_args(_args);
	}
}

bool Picviz::PVMappingProperties::operator==(const PVMappingProperties& org)
{
	// These properties are equal if and only if the same filter is used on the same index
	return (_mapping_filter == org._mapping_filter) && (_index == org._index);
}

void Picviz::PVMappingProperties::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("type", _type);
	so.attribute("mode", _mode);
	so.attribute("index", _index);
	so.attribute("group_key", _group_key);
	so.arguments("properties", _args);

	if (!so.is_writing()) {
		_is_uptodate = false;
		set_type(_type, _mode);
	}
}

void Picviz::PVMappingProperties::set_default_args(PVRush::PVAxisFormat const& axis)
{
	if (_args.size() == 0) {
		set_args(axis.get_args_mapping());
	}
}
