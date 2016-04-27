/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMapping.h>
#include <inendi/PVMappingProperties.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <inendi/PVRoot.h>

Inendi::PVMappingProperties::PVMappingProperties(PVRush::PVFormat const& format, PVCol idx)
{
	_index = idx;
	set_from_axis(format.get_axes().at(idx));
}

Inendi::PVMappingProperties::PVMappingProperties(PVRush::PVAxisFormat const& axis, PVCol idx)
{
	_index = idx;
	set_from_axis(axis);
}

void Inendi::PVMappingProperties::set_from_axis(PVRush::PVAxisFormat const& axis)
{
	set_from_axis(Inendi::PVAxis(axis));
}

void Inendi::PVMappingProperties::set_from_axis(Inendi::PVAxis const& axis)
{
	QString type = axis.get_type();
	QString mode = axis.get_mapping();
	PVCore::PVArgumentList args = axis.get_args_mapping();

	set_args(args);
	set_type(type, mode);
}

void Inendi::PVMappingProperties::set_type(QString const& type, QString const& mode)
{
	if (_is_uptodate && _type == type && _mode == mode) {
		return;
	}
	_type = type;
	_is_uptodate = false;
	set_mode(mode);
}

void Inendi::PVMappingProperties::set_args(PVCore::PVArgumentList const& args)
{
	if (!PVCore::comp_hash(args, _args)) {
		_is_uptodate = false;
	}
	if (_mapping_filter) {
		PVCore::PVArgumentList new_args = _mapping_filter->get_default_args();
		PVArgumentList_set_common_args_from(new_args, args);
		_mapping_filter->set_args(new_args);
		_args = new_args;
	} else {
		_args = args;
	}
}

void Inendi::PVMappingProperties::set_mode(QString const& mode)
{
	if (_is_uptodate && _mode == mode) {
		return;
	}
	_is_uptodate = false;
	_mode = mode;
	PVMappingFilter::p_type lib_filter =
	    LIB_CLASS(Inendi::PVMappingFilter)::get().get_class_by_name(_type + "_" + mode);

	_mapping_filter = lib_filter->clone<PVMappingFilter>();
	set_args(_args);
}

bool Inendi::PVMappingProperties::operator==(const PVMappingProperties& org)
{
	// These properties are equal if and only if the same filter is used on the
	// same index
	return (_mapping_filter == org._mapping_filter) && (_index == org._index);
}

void Inendi::PVMappingProperties::serialize(PVCore::PVSerializeObject& so,
                                            PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("type", _type);
	so.attribute("mode", _mode);
	so.attribute("index", _index);

	if (!so.is_writing()) {
		_is_uptodate = false;
		set_type(_type, _mode);
	}
	if (_mapping_filter) {
		so.arguments("properties", _args, _mapping_filter->get_default_args());
		if (!so.is_writing()) {
			_mapping_filter->set_args(_args);
		}
	}
}

void Inendi::PVMappingProperties::set_default_args(PVRush::PVAxisFormat const& axis)
{
	if (_args.size() == 0) {
		set_args(Inendi::PVAxis(axis).get_args_mapping());
	}
}
