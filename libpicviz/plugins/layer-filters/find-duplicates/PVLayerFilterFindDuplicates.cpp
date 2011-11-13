//! \file PVLayerFilterFindDuplicates.cpp
//! $Id: PVLayerFilterFindDuplicates.cpp 2526 2011-05-02 12:21:26Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include "PVLayerFilterFindDuplicates.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVCheckBoxType.h>
#include <picviz/PVView.h>


/******************************************************************************
 *
 * Picviz::PVLayerFilterFindDuplicates::PVLayerFilterFindDuplicates
 *
 *****************************************************************************/
Picviz::PVLayerFilterFindDuplicates::PVLayerFilterFindDuplicates(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterFindDuplicates, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterFindDuplicates)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterFindDuplicates)
{
	PVCore::PVArgumentList args;
	args["Axis"].setValue(PVCore::PVAxisIndexType(0));
	args["Select only non-duplicates"].setValue(PVCore::PVCheckBoxType(false));
	args["Select with one duplicate"].setValue(PVCore::PVCheckBoxType(true));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterFindDuplicates::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterFindDuplicates::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_id = _args["Axis"].value<PVCore::PVAxisIndexType>().get_original_index();
	bool non_duplicates = _args["Select only non-duplicates"].value<PVCore::PVCheckBoxType>().get_checked();
	bool with_one_duplicate = _args["Select with one duplicate"].value<PVCore::PVCheckBoxType>().get_checked();
	PVRow nb_lines = _view->get_qtnraw_parent().size();

	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();
	QHash<QString, PVRow> lines_duplicates;
	QHash<PVRow, bool> line_already_selected; // Hash storing if we already selected a duplicate or not (option with_one_duplicate)

	PVLOG_INFO("Select non duplicates:%d; with one duplicate only: %d\n", non_duplicates, with_one_duplicate);

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
				if ((line_already_selected[r]) || non_duplicates) {
					out.get_selection().set_line(r, non_duplicates ? false : true);
				} else {
					out.get_selection().set_line(r, true);
					line_already_selected.insert(r, true);
				}
			} else {
				out.get_selection().set_line(r, non_duplicates ? true : false);
			}
		}
	}

}

IMPL_FILTER(Picviz::PVLayerFilterFindDuplicates)
