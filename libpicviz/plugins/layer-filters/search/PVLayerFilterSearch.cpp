//! \file PVLayerFilterSearch.cpp
//! $Id: PVLayerFilterSearch.cpp 2526 2011-05-02 12:21:26Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include "PVLayerFilterSearch.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVEnumType.h>
#include <picviz/PVView.h>


#define INCLUDE_EXCLUDE_STR "Include or exclude pattern"
#define CASE_SENSITIVE_STR "Case sensitivity"
/******************************************************************************
 *
 * Picviz::PVLayerFilterSearch::PVLayerFilterSearch
 *
 *****************************************************************************/
Picviz::PVLayerFilterSearch::PVLayerFilterSearch(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterSearch, l);
	add_ctxt_menu_entry("Search for this value", &PVLayerFilterSearch::search_value_menu);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterSearch)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterSearch)
{
	PVCore::PVArgumentList args;
	args["Regular expression"] = QRegExp("(.*)");
	args["Axis"].setValue(PVCore::PVAxisIndexType(0));
	args[INCLUDE_EXCLUDE_STR].setValue(PVCore::PVEnumType(QStringList() << QString("include") << QString("exclude"), 0));
	args[CASE_SENSITIVE_STR].setValue(PVCore::PVEnumType(QStringList() << QString("Does not match case") << QString("Match case") , 0));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterSearch::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterSearch::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_id = _args["Axis"].value<PVCore::PVAxisIndexType>().get_original_index();
	QRegExp re = _args["Regular expression"].toRegExp();
	bool include = _args[INCLUDE_EXCLUDE_STR].value<PVCore::PVEnumType>().get_sel_index() == 0;
	bool case_match = _args[CASE_SENSITIVE_STR].value<PVCore::PVEnumType>().get_sel_index() == 1;
	re.setCaseSensitivity((Qt::CaseSensitivity) case_match);
	PVLOG_INFO("Apply filter search to axis %d with regexp %s.\n", axis_id, qPrintable(re.pattern()));

	PVRow nb_lines = _view->get_qtnraw_parent().get_nrows();

	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();
	
	for (PVRow r = 0; r < nb_lines; r++) {
		if (should_cancel()) {
			if (&in != &out) {
				out = in;
			}
			return;
		}

		if (_view->get_line_state_in_pre_filter_layer(r)) {
			PVRush::PVNraw::const_nraw_table_line nraw_r = nraw.get_row(r);
			bool sel = !((re.indexIn(nraw_r[axis_id].get_qstr()) != -1) ^ include);
			out.get_selection().set_line(r, sel);
		}
	}
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterSearch::search_value_menu
 *
 *****************************************************************************/
PVCore::PVArgumentList Picviz::PVLayerFilterSearch::search_value_menu(PVRow row, PVCol col, QString const& v)
{
	PVCore::PVArgumentList args = default_args();
	args["Regular expression"] = QRegExp(QRegExp::escape(v));
	args["Axis"].setValue(PVCore::PVAxisIndexType(col));
	return args;
}

IMPL_FILTER(Picviz::PVLayerFilterSearch)
