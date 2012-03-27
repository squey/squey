#include "PVLayerFilterSelectionSearch.h"
#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <picviz/PVView.h>

#include <omp.h>



#define ARG_NAME_AXIS "axis"
#define ARG_DESC_AXIS "Based on values of axis"
#define ARG_NAME_CASE "case"
#define ARG_DESC_CASE "Case sensitivity"
#define ARG_NAME_ENTIRE "entire"
#define ARG_DESC_ENTIRE "Match on"
#define ARG_NAME_INTERPRET "interpret"
#define ARG_DESC_INTERPRET "Interpret expressions as"


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
	args[PVCore::PVArgumentKey(ARG_NAME_AXIS, QObject::tr(ARG_DESC_AXIS))].setValue(PVCore::PVAxisIndexType(0));
	args[PVCore::PVArgumentKey(ARG_NAME_CASE, QObject::tr(ARG_DESC_CASE))].setValue(PVCore::PVEnumType(QStringList() << QString("Does not match case") << QString("Match case") , 0));
	args[PVCore::PVArgumentKey(ARG_NAME_ENTIRE, QObject::tr(ARG_DESC_ENTIRE))].setValue(PVCore::PVEnumType(QStringList() << QString("Part of the field") << QString("The entire field") , 0));
	args[PVCore::PVArgumentKey(ARG_NAME_INTERPRET, QObject::tr(ARG_DESC_INTERPRET))].setValue(PVCore::PVEnumType(QStringList() << QString("Plain text") << QString("Regular expressions") << QString("Wildcard") , 0));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterSelectionSearch::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterSelectionSearch::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_id = _args[ARG_NAME_AXIS].value<PVCore::PVAxisIndexType>().get_original_index();
	int interpret = _args[ARG_NAME_INTERPRET].value<PVCore::PVEnumType>().get_sel_index();
	bool case_match = _args[ARG_NAME_CASE].value<PVCore::PVEnumType>().get_sel_index() == 1;
	bool exact_match = _args[ARG_NAME_ENTIRE].value<PVCore::PVEnumType>().get_sel_index() == 1;
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

	Picviz::PVSelection out_sel_red[omp_get_max_threads()];
#pragma omp parallel
	{
		Picviz::PVSelection& out_sel = out_sel_red[omp_get_thread_num()];
#pragma omp parallel for
		for (PVRow r = 0; r < nb_lines; r++) {
			if (should_cancel()) {
				// Do nothing, so that this Open-MP loop will end very fast.
				continue;
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
				out_sel.set_line(r, true);
			}
		}
	}
	if (should_cancel()) {
		return;
	}

	// End of the reduction
	Picviz::PVSelection& out_sel = out.get_selection();
	for (int i = 0; i < omp_get_max_threads(); i++) {
		out_sel |= out_sel_red[i];
	}
}

QList<PVCore::PVArgumentKey> Picviz::PVLayerFilterSelectionSearch::get_args_keys_for_preset() const
{
	QList<PVCore::PVArgumentKey> keys = get_default_args().keys();
	keys.removeAll(ARG_NAME_AXIS);
	return keys;
}

PVCore::PVArgumentList Picviz::PVLayerFilterSelectionSearch::sel_axis_menu(PVRow /*row*/, PVCol col, QString const& /*v*/)
{
	PVCore::PVArgumentList args;
	args[ARG_NAME_AXIS].setValue(PVCore::PVAxisIndexType(col));
	return args;
}

IMPL_FILTER(Picviz::PVLayerFilterSelectionSearch)
