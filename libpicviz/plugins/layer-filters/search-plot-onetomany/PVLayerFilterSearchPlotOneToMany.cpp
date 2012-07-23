/**
 * \file PVLayerFilterSearchPlotOneToMany.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include "PVLayerFilterSearchPlotOneToMany.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVTextEditType.h>
#include <picviz/PVView.h>

#include <QList>
#include <QMap>
#include <QSpinBox>

#define ARG_NAME_NUMBER "number"
#define ARG_DESC_NUMBER "Number"
#define ARG_NAME_AXIS_FROM "axis_from"
#define ARG_DESC_AXIS_FROM "Axis From"
#define ARG_NAME_AXIS_TO "axis_to"
#define ARG_DESC_AXIS_TO "Axis To"

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
	args[PVCore::PVArgumentKey(ARG_NAME_NUMBER, QObject::tr(ARG_DESC_NUMBER))] = QString("2");
	args[PVCore::PVArgumentKey(ARG_NAME_AXIS_FROM, QObject::tr(ARG_DESC_AXIS_FROM))].setValue(PVCore::PVAxisIndexType(0));
	args[PVCore::PVArgumentKey(ARG_NAME_AXIS_TO, QObject::tr(ARG_DESC_AXIS_TO))].setValue(PVCore::PVAxisIndexType(0));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterSearchPlotOneToMany::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterSearchPlotOneToMany::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_from_id = _args[ARG_NAME_AXIS_FROM].value<PVCore::PVAxisIndexType>().get_original_index();
	int axis_to_id = _args[ARG_NAME_AXIS_TO].value<PVCore::PVAxisIndexType>().get_original_index();
	QString number = _args[ARG_NAME_NUMBER].toString();
	int greater_counter = number.toInt();

	PVRow nb_lines = _view->get_qtnraw_parent().get_nrows();

	const PVPlotted	*plotted = _view->get_parent<PVPlotted>();

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

QList<PVCore::PVArgumentKey> Picviz::PVLayerFilterSearchPlotOneToMany::get_args_keys_for_preset() const
{
	QList<PVCore::PVArgumentKey> keys = get_default_args().keys();
	keys.removeAll(ARG_NAME_AXIS_FROM);
	keys.removeAll(ARG_NAME_AXIS_TO);
	return keys;
}

IMPL_FILTER(Picviz::PVLayerFilterSearchPlotOneToMany)
