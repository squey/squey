#include "PVLayerFilterMultipleSearch.h"
#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <picviz/PVView.h>

#define INCLUDE_EXCLUDE_STR "Include or exclude pattern"
#define CASE_SENSITIVE_STR "Case sensitivity"
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
	args[PVCore::PVArgumentKey("exps", QObject::tr("Expressions"))].setValue(PVCore::PVPlainTextType());
	args[PVCore::PVArgumentKey("axis", QObject::tr("Axis"))].setValue(PVCore::PVAxisIndexType(0));
	args[PVCore::PVArgumentKey("include", QObject::tr(INCLUDE_EXCLUDE_STR))].setValue(PVCore::PVEnumType(QStringList() << QString("include") << QString("exclude"), 0));
	args[PVCore::PVArgumentKey("case", QObject::tr(CASE_SENSITIVE_STR))].setValue(PVCore::PVEnumType(QStringList() << QString("Does not match case") << QString("Match case") , 0));
	args[PVCore::PVArgumentKey("entire", QObject::tr("Match on"))].setValue(PVCore::PVEnumType(QStringList() << QString("Part of the field") << QString("The entire field") , 0));
	args[PVCore::PVArgumentKey("interpret", QObject::tr("Interpret expressions as"))].setValue(PVCore::PVEnumType(QStringList() << QString("Plain text") << QString("Regular expressions") << QString("Wildcard") , 0));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterMultipleSearch::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterMultipleSearch::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_id = _args["axis"].value<PVCore::PVAxisIndexType>().get_original_index();
	int interpret = _args["interpret"].value<PVCore::PVEnumType>().get_sel_index();
	bool include = _args["include"].value<PVCore::PVEnumType>().get_sel_index() == 0;
	bool case_match = _args["case"].value<PVCore::PVEnumType>().get_sel_index() == 1;
	bool exact_match = _args["entire"].value<PVCore::PVEnumType>().get_sel_index() == 1;
	bool is_rx = interpret >= 1;
	bool is_wildcard = interpret == 2;

	QString const& txt = _args["exps"].value<PVCore::PVPlainTextType>().get_text();
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

IMPL_FILTER(Picviz::PVLayerFilterMultipleSearch)
