//! \file PVLayerFilterFindNotDuplicates.cpp
//! $Id: PVLayerFilterFindNotDuplicates.cpp 2526 2011-05-02 12:21:26Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include "PVLayerFilterFindNotDuplicates.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVEnumType.h>
#include <picviz/PVView.h>


/******************************************************************************
 *
 * Picviz::PVLayerFilterFindNotDuplicates::PVLayerFilterFindNotDuplicates
 *
 *****************************************************************************/
Picviz::PVLayerFilterFindNotDuplicates::PVLayerFilterFindNotDuplicates(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterFindNotDuplicates, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterFindNotDuplicates)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterFindNotDuplicates)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("axis", QObject::tr("Axis")].setValue(PVCore::PVAxisIndexType(0));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterFindNotDuplicates::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterFindNotDuplicates::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_id = _args["axis"].value<PVCore::PVAxisIndexType>().get_original_index();
	PVRow nb_lines = _view->get_qtnraw_parent().size();

	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();
	QHash<QString, PVRow> lines_duplicates;

	// First round = We get all lines 
	for (PVRow r = 0; r < nb_lines; r++) {
		if (should_cancel()) {
			if (&in != &out) {
				out = in;
			}
			return;
		}

		if (_view->get_line_state_in_pre_filter_layer(r)) {
			PVRush::PVNraw::nraw_table_line const& nraw_r = nraw.at(r);
			QString value = nraw_r[axis_id];
			PVRow count = lines_duplicates[value]+1;
			lines_duplicates.insert(value, count);
		}
	}

	// Second round = We select all duplicates 
	for (PVRow r = 0; r < nb_lines; r++) {
		if (should_cancel()) {
			if (&in != &out) {
				out = in;
			}
			return;
		}

		if (_view->get_line_state_in_pre_filter_layer(r)) {
			PVRush::PVNraw::nraw_table_line const& nraw_r = nraw.at(r);
			QString value = nraw_r[axis_id];
			PVRow count = lines_duplicates[value];
			if (count > 1) {
				out.get_selection().set_line(r, false);
			} else {
				out.get_selection().set_line(r, true);
			}
		}
	}

}

IMPL_FILTER(Picviz::PVLayerFilterFindNotDuplicates)
