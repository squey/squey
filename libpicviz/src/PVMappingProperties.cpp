//! \file PVMappingProperties.cpp
//! $Id: PVMappingProperties.cpp 3062 2011-06-07 08:33:36Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVMappingProperties.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <picviz/PVRoot.h>

Picviz::PVMappingProperties::PVMappingProperties(PVRoot_p root, PVRush::PVFormat fmt, int idx)
{
	format = fmt;
	index = idx;

	QString type = format.get_axes().at(idx).get_type();
	QString mode = format.get_axes().at(idx).get_mapping();

	mapping_filter = LIB_CLASS(Picviz::PVMappingFilter)::get().get_class_by_name(type + "_" + mode);
	if (!mapping_filter) {
		PVLOG_ERROR("Mapping '%s' for type '%s' does not exist !\n", qPrintable(mode), qPrintable(type));
	}
}
