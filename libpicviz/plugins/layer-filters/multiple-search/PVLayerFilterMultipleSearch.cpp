/**
 * \file PVLayerFilterMultipleSearch.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVLayerFilterMultipleSearch.h"
#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <picviz/PVView.h>

#define ARG_NAME_EXPS "exps"
#define ARG_DESC_EXPS "Expressions"
#define ARG_NAME_AXIS "axis"
#define ARG_DESC_AXIS "Axis"
#define ARG_NAME_INCLUDE "include"
#define ARG_DESC_INCLUDE "Include or exclude pattern"
#define ARG_NAME_CASE "case"
#define ARG_DESC_CASE "Case sensitivity"
#define ARG_NAME_ENTIRE "entire"
#define ARG_DESC_ENTIRE "Match on"
#define ARG_NAME_INTERPRET "interpret"
#define ARG_DESC_INTERPRET "Interpret expressions as"

/******************************************************************************
 *
 * Picviz::PVLayerFilterMultipleSearch::PVLayerFilterMultipleSearch
 *
 *****************************************************************************/
Picviz::PVLayerFilterMultipleSearch::PVLayerFilterMultipleSearch(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterMultipleSearch, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterMultipleSearch)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterMultipleSearch)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey(ARG_NAME_EXPS, QObject::tr(ARG_DESC_EXPS))].setValue(PVCore::PVPlainTextType());
	args[PVCore::PVArgumentKey(ARG_NAME_AXIS, QObject::tr(ARG_DESC_AXIS))].setValue(PVCore::PVAxisIndexType(0));
	args[PVCore::PVArgumentKey(ARG_NAME_INCLUDE, QObject::tr(ARG_DESC_INCLUDE))].setValue(PVCore::PVEnumType(QStringList() << QString("include") << QString("exclude"), 0));
	args[PVCore::PVArgumentKey(ARG_NAME_CASE, QObject::tr(ARG_DESC_CASE))].setValue(PVCore::PVEnumType(QStringList() << QString("Does not match case") << QString("Match case") , 0));
	args[PVCore::PVArgumentKey(ARG_NAME_ENTIRE, QObject::tr(ARG_DESC_ENTIRE))].setValue(PVCore::PVEnumType(QStringList() << QString("Part of the field") << QString("The entire field") , 0));
	args[PVCore::PVArgumentKey(ARG_NAME_INTERPRET, QObject::tr(ARG_DESC_INTERPRET))].setValue(PVCore::PVEnumType(QStringList() << QString("Plain text") << QString("Regular expressions") << QString("Wildcard") , 0));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterMultipleSearch::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterMultipleSearch::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_id = _args[ARG_NAME_AXIS].value<PVCore::PVAxisIndexType>().get_original_index();
	int interpret = _args[ARG_NAME_INTERPRET].value<PVCore::PVEnumType>().get_sel_index();
	bool include = _args[ARG_NAME_INCLUDE].value<PVCore::PVEnumType>().get_sel_index() == 0;
	bool case_match = _args[ARG_NAME_CASE].value<PVCore::PVEnumType>().get_sel_index() == 1;
	bool exact_match = _args[ARG_NAME_ENTIRE].value<PVCore::PVEnumType>().get_sel_index() == 1;
	bool is_rx = interpret >= 1;
	bool is_wildcard = interpret == 2;

	QString const& txt = _args[ARG_NAME_EXPS].value<PVCore::PVPlainTextType>().get_text();
	QStringList exps = txt.split("\n");
	std::vector<QRegExp> rxs;
	if (is_rx) {
		rxs.reserve(exps.size());
		for (int i = 0; i < exps.size(); i++) {
			QString pattern = exps.at(i).trimmed();
			if (!pattern.isEmpty()) {
				QRegExp rx;
				rx.setPattern(exps.at(i).trimmed());
				rx.setCaseSensitivity((Qt::CaseSensitivity) case_match);
				if (is_wildcard) {
					rx.setPatternSyntax(QRegExp::WildcardUnix);
				}
				rxs.push_back(rx);
			}
		}
	}
	else {
		for (int i = 0; i < exps.size(); i++) {
			QString& str = exps[i];
			str = str.trimmed();
		}
	}

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
			bool sel = false;
			if (is_rx) {
				for (size_t i = 0; i < rxs.size(); i++) {
					QRegExp& rx = rxs[i];
					QString str(nraw_r[axis_id].get_qstr());
					if (exact_match) {
						if (rx.exactMatch(str)) {
							sel = true;
							break;
						}
					}
					else
					if (rx.indexIn(str) != -1) {
						sel = true;
						break;
					}
				}
			}
			else {
				QString str = nraw_r[axis_id].get_qstr();
				for (int i = 0; i < exps.size(); i++) {
					QString const& exp = exps.at(i);
					if (exp.isEmpty()) {
					   continue;
					}
					if (exact_match) {
						if (str.compare(exp, (Qt::CaseSensitivity) case_match) == 0) {
							sel = true;
							break;
						}
					}
					else
					if (str.contains(exp, (Qt::CaseSensitivity) case_match)) {
						sel = true;
						break;
					}
				}
			}

			sel = !(sel ^ include);
			out.get_selection().set_line(r, sel);
		}
	}
}

QList<PVCore::PVArgumentKey> Picviz::PVLayerFilterMultipleSearch::get_args_keys_for_preset() const
{
	QList<PVCore::PVArgumentKey> keys = get_default_args().keys();
	keys.removeAll(ARG_NAME_AXIS);
	return keys;
}

IMPL_FILTER(Picviz::PVLayerFilterMultipleSearch)
