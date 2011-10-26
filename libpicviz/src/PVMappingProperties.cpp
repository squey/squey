//! \file PVMappingProperties.cpp
//! $Id: PVMappingProperties.cpp 3062 2011-06-07 08:33:36Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVMapping.h>
#include <picviz/PVMappingProperties.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <picviz/PVRoot.h>

Picviz::PVMappingProperties::PVMappingProperties(PVRush::PVFormat const& format, int idx)
{
	_index = idx;

	QString type = format.get_axes().at(idx).get_type();
	QString mode = format.get_axes().at(idx).get_mapping();
	QString group = format.get_axes().at(idx).get_group();

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

void Picviz::PVMappingProperties::set_mode(QString const& mode)
{
	if (_is_uptodate && _mode == mode) {
		return;
	}
	_is_uptodate = false;
	_mode = mode;
	_mapping_filter = LIB_CLASS(Picviz::PVMappingFilter)::get().get_class_by_name(_type + "_" + mode);
	if (!_mapping_filter) {
		PVLOG_ERROR("Mapping '%s' for type '%s' does not exist !\n", qPrintable(mode), qPrintable(_type));
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

	if (!so.is_writing()) {
		_is_uptodate = false;
		set_type(_type, _mode);
	}
}
