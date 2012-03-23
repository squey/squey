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

#define ARG_NAME_AXIS "Axis"
#define ARG_NAME_REG_EXP "Regular expression"
#define ARG_NAME_INCLUDE "Include or exclude pattern"
#define ARG_NAME_CASE "Case sensitivity"

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
	args[ARG_NAME_REG_EXP] = QRegExp("(.*)");
	args[ARG_NAME_AXIS].setValue(PVCore::PVAxisIndexType(0));
	args[ARG_NAME_INCLUDE].setValue(PVCore::PVEnumType(QStringList() << QString("include") << QString("exclude"), 0));
	args[ARG_NAME_CASE].setValue(PVCore::PVEnumType(QStringList() << QString("Does not match case") << QString("Match case") , 0));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterSearch::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterSearch::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_id = _args[ARG_NAME_AXIS].value<PVCore::PVAxisIndexType>().get_original_index();
	QRegExp re = _args[ARG_NAME_REG_EXP].toRegExp();
	bool include = _args[ARG_NAME_INCLUDE].value<PVCore::PVEnumType>().get_sel_index() == 0;
	bool case_match = _args[ARG_NAME_CASE].value<PVCore::PVEnumType>().get_sel_index() == 1;
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

QList<PVCore::PVArgumentKey> Picviz::PVLayerFilterSearch::get_args_keys_for_preset() const
{
	QList<PVCore::PVArgumentKey> keys = get_args_for_preset().keys();
	keys.removeAll(ARG_NAME_AXIS);
	return keys;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterSearch::search_value_menu
 *
 *****************************************************************************/
PVCore::PVArgumentList Picviz::PVLayerFilterSearch::search_value_menu(PVRow row, PVCol col, QString const& v)
{
	PVCore::PVArgumentList args = default_args();
	args[ARG_NAME_REG_EXP] = QRegExp(QRegExp::escape(v));
	args[ARG_NAME_AXIS].setValue(PVCore::PVAxisIndexType(col));
	return args;
}

IMPL_FILTER(Picviz::PVLayerFilterSearch)
