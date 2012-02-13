//! \file PVLayerFilterSearchPlotOneToMany.cpp
//! $Id: PVLayerFilterSearch.cpp 2526 2011-05-02 12:21:26Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2012
//! Copyright (C) Philippe Saadé 2009-2012
//! Copyright (C) Picviz Labs 2012

#include "PVLayerFilterSearchPlotOneToMany.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVTextEditType.h>
#include <picviz/PVView.h>

#include <QList>
#include <QMap>
#include <QSpinBox>


#define INCLUDE_EXCLUDE_STR "Include or exclude pattern"
#define CASE_SENSITIVE_STR "Case sensitivity"
/******************************************************************************
 *
 * Picviz::PVLayerFilterSearchPlotOneToMany::PVLayerFilterSearchPlotOneToMany
 *
 *****************************************************************************/
Picviz::PVLayerFilterSearchPlotOneToMany::PVLayerFilterSearchPlotOneToMany(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterSearchPlotOneToMany, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterSearchPlotOneToMany)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterSearchPlotOneToMany)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("number", QObject::tr("Number"))] = QString("2");
	args[PVCore::PVArgumentKey("axis_from", QObject::tr("Axis From"))].setValue(PVCore::PVAxisIndexType(0));
	args[PVCore::PVArgumentKey("axis_to", QObject::tr("Axis To"))].setValue(PVCore::PVAxisIndexType(0));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterSearchPlotOneToMany::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterSearchPlotOneToMany::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_from_id = _args["axis_from"].value<PVCore::PVAxisIndexType>().get_original_index();
	int axis_to_id = _args["axis_to"].value<PVCore::PVAxisIndexType>().get_original_index();
	QString number = _args["number"].toString();
	int greater_counter = number.toInt();

	PVRow nb_lines = _view->get_qtnraw_parent().get_nrows();

	const PVPlotted	*plotted = _view->get_plotted_parent();

	QMap <float, QList<float> > plotted_groups;


	out.get_selection().select_none();


	for (PVRow r = 0; r < nb_lines; r++) {
		if (should_cancel()) {
			if (&in != &out) {
				out = in;
			}
			return;
		}

		// First round: We group
		if (_view->get_line_state_in_pre_filter_layer(r)) {
			float value_from = plotted->get_value((PVRow)axis_from_id, r);
			float value_to = plotted->get_value((PVRow)axis_to_id, r);

			plotted_groups[value_from] << value_to;
		}

		// Second round: We unroll the group to select
		if (_view->get_line_state_in_pre_filter_layer(r)) {
			float value_from = plotted->get_value((PVRow)axis_from_id, r);
			QList<float> to_values = plotted_groups.value(value_from);
			if (to_values.count() >= greater_counter) {
				out.get_selection().set_line(r, true);
			}
		}
		

	}
}


IMPL_FILTER(Picviz::PVLayerFilterSearchPlotOneToMany)
