//! \file PVLayerFilterFindDuplicates.cpp
//! $Id: PVLayerFilterFindDuplicates.cpp 2526 2011-05-02 12:21:26Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include "PVLayerFilterFindDuplicates.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <picviz/PVView.h>

#define ARG_NAME_AXIS "Axis"
#define ARG_NAME_SELECT_ONLY_DUP "select-only-dup"
#define ARG_DESC_SELECT_ONLY_DUP "Select only non-duplicates"
#define ARG_NAME_SELECT_ONLY_ONE_DUP "select-only-one-dup"
#define ARG_DESC_SELECT_ONLY_ONE_DUP "Select with one duplicate"

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
	args[ARG_NAME_AXIS].setValue(PVCore::PVAxisIndexType(0));
	args[PVCore::PVArgumentKey(ARG_NAME_SELECT_ONLY_DUP, ARG_DESC_SELECT_ONLY_DUP)].setValue<bool>(false);
	args[PVCore::PVArgumentKey(ARG_NAME_SELECT_ONLY_ONE_DUP, ARG_DESC_SELECT_ONLY_ONE_DUP)].setValue<bool>(true);
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterFindDuplicates::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterFindDuplicates::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_id = _args[ARG_NAME_AXIS].value<PVCore::PVAxisIndexType>().get_original_index();
	bool non_duplicates = _args[ARG_NAME_SELECT_ONLY_DUP].toBool();
	bool with_one_duplicate = _args[ARG_NAME_SELECT_ONLY_ONE_DUP].toBool();
	PVRow nb_lines = _view->get_qtnraw_parent().get_nrows();

	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();
	QHash<QString, PVRow> lines_duplicates;
	QHash<PVRow, bool> line_already_selected; // Hash storing if we already selected a duplicate or not (option with_one_duplicate)
	QHash<QString, bool> value_already_selected;

	// PVLOG_INFO("Select non duplicates:%d; with one duplicate only: %d\n", non_duplicates, with_one_duplicate);

	// First round = We get all lines 
	for (PVRow r = 0; r < nb_lines; r++) {
		if (should_cancel()) {
			if (&in != &out) {
				out = in;
			}
			return;
		}

		if (_view->get_line_state_in_pre_filter_layer(r)) {
			QString value = nraw.at(r, axis_id).get_qstr();
			PVRow count = lines_duplicates[value] + 1;
			// PVLOG_INFO("We insert dup info for value '%s', count '%d'\n", qPrintable(value),count);
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
			QString value = nraw.at(r, axis_id).get_qstr();
			PVRow count = lines_duplicates[value];
			if (count > 1) {
				if (non_duplicates) {
					out.get_selection().set_line(r, false);
				} else {
					if (with_one_duplicate) {
						if (value_already_selected[value]) {
							out.get_selection().set_line(r, false);
						} else {
							out.get_selection().set_line(r, true);
							value_already_selected.insert(value, true);
						}
					}
				}
			} else {
				out.get_selection().set_line(r, true);
			}
		}
	}

}

QList<PVCore::PVArgumentKey> Picviz::PVLayerFilterFindDuplicates::get_args_keys_for_preset() const
{
	QList<PVCore::PVArgumentKey> keys = get_args_for_preset().keys();
	keys.removeAll(ARG_NAME_AXIS);
	return keys;
}

IMPL_FILTER(Picviz::PVLayerFilterFindDuplicates)
