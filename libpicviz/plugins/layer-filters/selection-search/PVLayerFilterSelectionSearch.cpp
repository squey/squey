#include "PVLayerFilterSelectionSearch.h"
#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <picviz/PVView.h>

#define INCLUDE_EXCLUDE_STR "Include or exclude pattern"
#define CASE_SENSITIVE_STR "Case sensitivity"
/******************************************************************************
 *
 * Picviz::PVLayerFilterSelectionSearch::PVLayerFilterSelectionSearch
 *
 *****************************************************************************/
Picviz::PVLayerFilterSelectionSearch::PVLayerFilterSelectionSearch(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterSelectionSearch, l);
	add_ctxt_menu_entry(QObject::tr("Selection-based search on this axis"), &PVLayerFilterSelectionSearch::sel_axis_menu);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterSelectionSearch)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterSelectionSearch)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("axis", QObject::tr("Based on values of axis"))].setValue(PVCore::PVAxisIndexType(0));
	args[PVCore::PVArgumentKey("case", QObject::tr(CASE_SENSITIVE_STR))].setValue(PVCore::PVEnumType(QStringList() << QString("Does not match case") << QString("Match case") , 0));
	args[PVCore::PVArgumentKey("entire", QObject::tr("Match on"))].setValue(PVCore::PVEnumType(QStringList() << QString("Part of the field") << QString("The entire field") , 0));
	args[PVCore::PVArgumentKey("interpret", QObject::tr("Interpret expressions as"))].setValue(PVCore::PVEnumType(QStringList() << QString("Plain text") << QString("Regular expressions") << QString("Wildcard") , 0));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterSelectionSearch::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterSelectionSearch::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_id = _args["axis"].value<PVCore::PVAxisIndexType>().get_original_index();
	int interpret = _args["interpret"].value<PVCore::PVEnumType>().get_sel_index();
	bool case_match = _args["case"].value<PVCore::PVEnumType>().get_sel_index() == 1;
	bool exact_match = _args["entire"].value<PVCore::PVEnumType>().get_sel_index() == 1;
	bool is_rx = interpret >= 1;
	bool is_wildcard = interpret == 2;

	PVRush::PVNraw::nraw_table_axis nraw_axis = _view->get_rushnraw_parent().get_col(axis_id);
	PVRow nb_lines = nraw_axis.size();

	QList<QStringMatcher> exps;
	std::vector<QRegExp> rxs;
	QSet<QString> exps_exact;
	if (is_rx) {
		rxs.reserve(nb_lines);
		for (PVRow i = 0; i < nb_lines; i++) {
			if (_view->get_line_state_in_pre_filter_layer(i)) {
				QString pattern(nraw_axis.at(i).get_qstr().trimmed());
				if (pattern.isEmpty()) {
					continue;
				}
				{
					std::vector<QRegExp>::const_iterator it;
					bool found = false;
					for (it = rxs.begin(); it != rxs.end(); it++) {
						if (it->pattern() == pattern) {
							found = true;
							break;
						}
					}
					if (found) {
						continue;
					}
				}
				QRegExp rx;
				rx.setPattern(pattern);
				rx.setCaseSensitivity((Qt::CaseSensitivity) case_match);
				if (is_wildcard) {
					rx.setPatternSyntax(QRegExp::WildcardUnix);
				}
				rxs.push_back(rx);
			}
		}
	}
	else
	if (exact_match) {
		for (PVRow i = 0; i < nb_lines; i++) {
			if (_view->get_line_state_in_pre_filter_layer(i)) {
				QString str(nraw_axis.at(i).get_qstr());
				if (!str.isEmpty()) {
					exps_exact << str;
				}
			}
		}
	}
	else
	for (PVRow i = 0; i < nb_lines; i++) {
		if (_view->get_line_state_in_pre_filter_layer(i)) {
			bool found = false;
			QString str(nraw_axis.at(i).get_qstr().trimmed());
			if (str.isEmpty()) {
				continue;
			}
			foreach(QStringMatcher const& m, exps) {
				if (m.pattern() == str) {
					found = true;
					break;
				}
			}
			if (!found) {
				exps << QStringMatcher(str, (Qt::CaseSensitivity) case_match);
			}
		}
	}

	for (PVRow r = 0; r < nb_lines; r++) {
		if (should_cancel()) {
			if (&in != &out) {
				out = in;
			}
			return;
		}

		bool sel = false;
		if (is_rx) {
			for (size_t i = 0; i < rxs.size(); i++) {
				QRegExp& rx = rxs[i];
				QString str(nraw_axis.at(r).get_qstr());
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
		else
		if (exact_match) {
			QString str(nraw_axis.at(r).get_qstr());
			if (case_match) {
				foreach(QString const& p, exps_exact) {
					if (str == p) {
						sel = true;
						break;
					}
				}
			}
			else {
				foreach(QString const& p, exps_exact) {
					if (str.compare(p, Qt::CaseInsensitive) == 0) {
						sel = true;
						break;
					}
				}
			}
		}
		else {
			QString str(nraw_axis.at(r).get_qstr());
			foreach(QStringMatcher const& exp, exps) {
				if (exp.indexIn(str) != -1) {
					sel = true;
					break;
				}
			}
		}

		if (sel) {
			out.get_selection().set_line(r, true);
		}
	}
}

PVCore::PVArgumentList Picviz::PVLayerFilterSelectionSearch::sel_axis_menu(PVRow /*row*/, PVCol col, QString const& /*v*/)
{
	PVCore::PVArgumentList args;
	args["axis"].setValue(PVCore::PVAxisIndexType(col));
	return args;
}

IMPL_FILTER(Picviz::PVLayerFilterSelectionSearch)
