//! \file PVMappingProperties.cpp
//! $Id: PVMappingProperties.cpp 3062 2011-06-07 08:33:36Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVMapping.h>
#include <picviz/PVMappingProperties.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <picviz/PVRoot.h>

Picviz::PVMappingProperties::PVMappingProperties(PVMapping const& parent, PVRush::PVFormat const& format, int idx):
	_src_parent(parent.get_source_parent())
{
	_index = idx;

	_type = format.get_axes().at(idx).get_type();
	QString mode = format.get_axes().at(idx).get_mapping();
	QString group = format.get_axes().at(idx).get_group();

	set_mode(mode);

	if (!group.isEmpty()) {
		_group_key = group + "_" + _type;
	}
}

void Picviz::PVMappingProperties::set_mode(QString const& mode)
{
	_mapping_filter = LIB_CLASS(Picviz::PVMappingFilter)::get().get_class_by_name(_type + "_" + mode);
	if (!_mapping_filter) {
		PVLOG_ERROR("Mapping '%s' for type '%s' does not exist !\n", qPrintable(mode), qPrintable(_type));
	}
}

bool Picviz::PVMappingProperties::operator==(const PVMappingProperties& org)
{
	// These properties are equal if and only if the same filter is used on the same index with the same parent
	return (_mapping_filter == org._mapping_filter) && (_index == org._index) && (_src_parent == org._src_parent);
}
