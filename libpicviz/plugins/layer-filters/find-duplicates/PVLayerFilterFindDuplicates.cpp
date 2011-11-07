//! \file PVLayerFilterFindDuplicates.cpp
//! $Id: PVLayerFilterFindDuplicates.cpp 2526 2011-05-02 12:21:26Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include "PVLayerFilterFindDuplicates.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVEnumType.h>
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
	add_ctxt_menu_entry("Find duplicates for this value", &PVLayerFilterFindDuplicates::search_value_menu);
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
				out.get_selection().set_line(r, true);
			} else {
				out.get_selection().set_line(r, false);
			}
		}
	}

}

/******************************************************************************
 *
 * Picviz::PVLayerFilterFindDuplicates::search_value_menu
 *
 *****************************************************************************/
PVCore::PVArgumentList Picviz::PVLayerFilterFindDuplicates::search_value_menu(PVRow row, PVCol col, QString const& v)
{
	PVCore::PVArgumentList args = default_args();
	args["Regular expression"] = QRegExp(QRegExp::escape(v));
	args["Axis"].setValue(PVCore::PVAxisIndexType(col));
	return args;
}

IMPL_FILTER(Picviz::PVLayerFilterFindDuplicates)
