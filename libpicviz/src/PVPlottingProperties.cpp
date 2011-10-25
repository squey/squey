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
Picviz::PVPlottingProperties::PVPlottingProperties(PVPlotting const& parent, PVRush::PVFormat const& format, int idx):
	_parent(&parent)
{
	_index = idx;

	_type = format.get_axes().at(idx).get_type();
	QString mode = format.get_axes().at(idx).get_plotting();

	set_mode(mode);
}

void Picviz::PVPlottingProperties::set_mode(QString const& mode)
{
	_plotting_filter = LIB_CLASS(PVPlottingFilter)::get().get_class_by_name(_type + "_" + mode);
	if (!_plotting_filter) {
		PVLOG_ERROR("Plotting mode '%s' for type '%s' does not exist !\n", qPrintable(mode), qPrintable(_type));
	}
}

bool Picviz::PVPlottingProperties::operator==(PVPlottingProperties const& org)
{
	return (_plotting_filter == org._plotting_filter) && (_parent == org._parent) && (_index == org._index);
}
