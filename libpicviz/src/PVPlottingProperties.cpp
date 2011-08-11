//! \file PVPlottingProperties.cpp
//! $Id: PVPlottingProperties.cpp 3062 2011-06-07 08:33:36Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVPlottingProperties.h>
#include <picviz/PVRoot.h>
#include <picviz/PVPlottingFilter.h>

#include <pvkernel/core/PVClassLibrary.h>

/******************************************************************************
 *
 * Picviz::PVPlottingProperties::PVPlottingProperties
 *
 *****************************************************************************/
Picviz::PVPlottingProperties::PVPlottingProperties(PVRoot_p root, PVRush::PVFormat fmt, int idx)
{
	format = fmt;
	index = idx;

	QString type = format.get_axes().at(idx).get_type();
	QString mode = format.get_axes().at(idx).get_plotting();

	plotting_filter = LIB_CLASS(PVPlottingFilter)::get().get_class_by_name(type + "_" + mode);
	if (!plotting_filter) {
		PVLOG_ERROR("Plotting mode '%s' for type '%s' does not exist !\n", qPrintable(mode), qPrintable(type));
	}
}
