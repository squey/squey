/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVLayerFilterMultipleSearch.h"

#include <inendi/PVView.h>

#include <pvkernel/rush/PVNraw.h>

#include <pvkernel/core/inendi_bench.h>
#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>

#include <pvcop/db/algo.h>
#include <pvcop/core/algo/selection.h>

#include <locale.h>

#include <tbb/enumerable_thread_specific.h>

#include <pcrecpp.h>

#include <QMessageBox>

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
#define ARG_NAME_TYPE "type"
#define ARG_DESC_TYPE "Search in"

/******************************************************************************
 *
 * Inendi::PVLayerFilterMultipleSearch::PVLayerFilterMultipleSearch
 *
 *****************************************************************************/
Inendi::PVLayerFilterMultipleSearch::PVLayerFilterMultipleSearch(PVCore::PVArgumentList const& l)
    : PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterMultipleSearch, l);
	add_ctxt_menu_entry("Search for this value", &PVLayerFilterMultipleSearch::search_value_menu);
	add_ctxt_menu_entry("Search using this value...",
	                    &PVLayerFilterMultipleSearch::search_using_value_menu);
	add_ctxt_menu_entry("Search for...", &PVLayerFilterMultipleSearch::search_menu);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Inendi::PVLayerFilterMultipleSearch)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Inendi::PVLayerFilterMultipleSearch)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey(ARG_NAME_EXPS, QObject::tr(ARG_DESC_EXPS))].setValue(
	    PVCore::PVPlainTextType());
	args[PVCore::PVArgumentKey(ARG_NAME_AXIS, QObject::tr(ARG_DESC_AXIS))].setValue(
	    PVCore::PVOriginalAxisIndexType(0));
	args[PVCore::PVArgumentKey(ARG_NAME_INCLUDE, QObject::tr(ARG_DESC_INCLUDE))].setValue(
	    PVCore::PVEnumType(QStringList() << QString("include") << QString("exclude"), 0));
	args[PVCore::PVArgumentKey(ARG_NAME_CASE, QObject::tr(ARG_DESC_CASE))].setValue(
	    PVCore::PVEnumType(QStringList() << QString("Does not match case") << QString("Match case"),
	                       1));
	args[PVCore::PVArgumentKey(ARG_NAME_ENTIRE, QObject::tr(ARG_DESC_ENTIRE))].setValue(
	    PVCore::PVEnumType(
	        QStringList() << QString("Part of the field") << QString("The entire field"), 1));
	args[PVCore::PVArgumentKey(ARG_NAME_INTERPRET, QObject::tr(ARG_DESC_INTERPRET))].setValue(
	    PVCore::PVEnumType(QStringList() << QString("Plain text") << QString("Regular expressions"),
	                       0));
	args[PVCore::PVArgumentKey(ARG_NAME_TYPE, QObject::tr(ARG_DESC_TYPE))].setValue(
	    PVCore::PVEnumType(QStringList() << QString("Valid values") << QString("Invalid values")
	                                     << QString("All values"),
	                       0));

	return args;
}

enum ESearchOptions {
	NONE = 0,
	REGULAR_EXPRESSION = 1 << 0,
	EXACT_MATCH = 1 << 1,
	CASE_INSENSITIVE = 1 << 2
};

enum EType { VALID = 1 << 0, INVALID = 1 << 1, EMPTY = 1 << 2 };

/******************************************************************************
 *
 * Inendi::PVLayerFilterMultipleSearch::operator()
 *
 *****************************************************************************/

void Inendi::PVLayerFilterMultipleSearch::operator()(PVLayer const& in, PVLayer& out)
{
	int axis_id =
	    _args[ARG_NAME_AXIS].value<PVCore::PVOriginalAxisIndexType>().get_original_index();
	int interpret = _args[ARG_NAME_INTERPRET].value<PVCore::PVEnumType>().get_sel_index();
	bool include = _args[ARG_NAME_INCLUDE].value<PVCore::PVEnumType>().get_sel_index() == 0;
	bool case_match = _args[ARG_NAME_CASE].value<PVCore::PVEnumType>().get_sel_index() == 1;
	bool exact_match = _args[ARG_NAME_ENTIRE].value<PVCore::PVEnumType>().get_sel_index() == 1;
	bool is_rx = interpret >= 1;

	int t = _args[ARG_NAME_TYPE].value<PVCore::PVEnumType>().get_sel_index();

	size_t type = (t + 1);

	size_t opts = ((REGULAR_EXPRESSION * is_rx) | (EXACT_MATCH * exact_match) |
	               (CASE_INSENSITIVE * (not case_match)));

	const QString& txt = _args[ARG_NAME_EXPS].value<PVCore::PVPlainTextType>().get_text();

	// Remove last carriage return if present otherwise we would search for empty strings as well
	QStringList exps = (txt.right(1) == "\n" ? txt.left(txt.size() - 1) : txt).split("\n");

	std::vector<std::string> exps_utf8;
	exps_utf8.resize(exps.size());

	bool search_empty_values = false;
#pragma omp parallel for
	for (int i = 0; i < exps.size(); i++) {
		QString const& str = exps[i];
		if (!str.isEmpty()) {
			exps_utf8[i] = str.toUtf8().constData();
		} else {
			search_empty_values = true;
		}
	}

	type |= (EMPTY * search_empty_values);

	PVRush::PVNraw const& nraw = _view->get_rushnraw_parent();
	PVSelection& out_sel = out.get_selection();

	const pvcop::db::array& column = nraw.collection().column(axis_id);

	BENCH_START(subselect);

	out_sel.select_none();

	switch (opts) {
	case (EXACT_MATCH): {
		try {
			pvcop::db::array converted_array = pvcop::db::algo::to_array(column, exps_utf8);
			if (type & VALID) {
				search_values(axis_id, converted_array, in.get_selection(), out_sel);
			}
			if (type & EMPTY) {
				nraw.empty_values_search(axis_id, in.get_selection(), out_sel);
			}
		} catch (pvcop::db::exception::partially_converted_error& e) {
			if ((type & VALID) && e.incomplete_array().size() > 0) {
				search_values(axis_id, e.incomplete_array(), in.get_selection(), out_sel);
			}
			if (type & INVALID) {
				const auto& inv = e.bad_values();
				nraw.unconvertable_values_search(axis_id, inv, in.get_selection(), out_sel);
			}
			if (type & EMPTY) {
				nraw.empty_values_search(axis_id, in.get_selection(), out_sel);
			}

			// Propagate exception if needed
			if (not((type & INVALID) || (type & EMPTY)) &&
			    std::string(nraw.collection().formatter(axis_id)->name()) != "string") {
				_unconverted_values = e.bad_values();
				throw PVLayerFilter::error(); // we should maybe not throw through plugin API and
				                              // set a flag instead...
			}
		}
	} break;
	case (EXACT_MATCH | CASE_INSENSITIVE): {
		auto predicate = [](const std::string& array_value, const std::string& exp_value) {
			return strcasecmp(array_value.c_str(), exp_value.c_str()) == 0;
		};
		search_values_if(axis_id, exps_utf8, type, in.get_selection(), out_sel, predicate);
	} break;
	case (EXACT_MATCH | REGULAR_EXPRESSION): {
		auto predicate = [](const std::string& array_value, const std::string& exp_value) {
			pcrecpp::RE re(exp_value.c_str(), PCRE_UTF8);
			return re.FullMatch(pcrecpp::StringPiece(array_value.c_str(), array_value.size()));
		};
		search_values_if(axis_id, exps_utf8, type, in.get_selection(), out_sel, predicate);
	} break;
	case (EXACT_MATCH | REGULAR_EXPRESSION | CASE_INSENSITIVE): {
		auto predicate = [](const std::string& array_value, const std::string& exp_value) {
			pcrecpp::RE re(exp_value.c_str(), PCRE_UTF8 | PCRE_CASELESS);
			return re.FullMatch(pcrecpp::StringPiece(array_value.c_str(), array_value.size()));
		};
		search_values_if(axis_id, exps_utf8, type, in.get_selection(), out_sel, predicate);
	} break;
	case (REGULAR_EXPRESSION): {
		auto predicate = [](const std::string& array_value, const std::string& exp_value) {
			pcrecpp::RE re(exp_value.c_str(), PCRE_UTF8);
			return re.PartialMatch(pcrecpp::StringPiece(array_value.c_str(), array_value.size()));
		};
		search_values_if(axis_id, exps_utf8, type, in.get_selection(), out_sel, predicate);
	} break;
	case (REGULAR_EXPRESSION | CASE_INSENSITIVE): {
		auto predicate = [](const std::string& array_value, const std::string& exp_value) {
			pcrecpp::RE re(exp_value.c_str(), PCRE_UTF8 | PCRE_CASELESS);
			return re.PartialMatch(pcrecpp::StringPiece(array_value.c_str(), array_value.size()));
		};
		search_values_if(axis_id, exps_utf8, type, in.get_selection(), out_sel, predicate);
	} break;
	case (CASE_INSENSITIVE): {
		auto predicate = [](const std::string& array_value, const std::string& exp_value) {
			return strcasestr(array_value.c_str(), exp_value.c_str()) != nullptr;
		};
		search_values_if(axis_id, exps_utf8, type, in.get_selection(), out_sel, predicate);
	} break;
	case (NONE): {
		auto predicate = [](const std::string& array_value, const std::string& exp_value) {
			return strstr(array_value.c_str(), exp_value.c_str()) != nullptr;
		};
		search_values_if(axis_id, exps_utf8, type, in.get_selection(), out_sel, predicate);
	} break;
	default: {
		assert(false);
	} break;
	}

	BENCH_END(subselect, "subselect", 1, 1, 1, 1);

	if (not include) {
		// invert selection
		BENCH_START(invert_selection);
		pvcop::core::algo::invert_selection(out_sel);
		out_sel &= in.get_selection();
		BENCH_END(invert_selection, "invert_selection", 1, 1, 1, 1);
	}
}

void Inendi::PVLayerFilterMultipleSearch::remove_default_invalid_values(
    PVCol col_idx, const Inendi::PVSelection& in_sel, Inendi::PVSelection& out_sel) const
{
	PVRush::PVNraw const& nraw = _view->get_rushnraw_parent();
	const pvcop::db::array& column = nraw.collection().column(col_idx);

	pvcop::db::array empty_array(column.type(), 1, true);

	Inendi::PVSelection ambiguous_values_sel(column.size());
	ambiguous_values_sel.select_none();
	pvcop::db::algo::subselect(column, empty_array, in_sel, ambiguous_values_sel);

	PVSelection tmp_sel(column.size());
	tmp_sel.select_none();
	nraw.unconvertable_values_search(col_idx, ambiguous_values_sel, tmp_sel);

	out_sel &= ~tmp_sel;

	tmp_sel.select_none();
	nraw.empty_values_search(col_idx, ambiguous_values_sel & ~tmp_sel, tmp_sel);

	out_sel &= ~tmp_sel;
}

void Inendi::PVLayerFilterMultipleSearch::search_values(PVCol col_idx,
                                                        const pvcop::db::array& search_array,
                                                        const PVSelection& in_sel,
                                                        Inendi::PVSelection& out_sel) const
{
	PVRush::PVNraw const& nraw = _view->get_rushnraw_parent();
	const pvcop::db::array& column = nraw.collection().column(col_idx);

	pvcop::db::algo::subselect(column, search_array, in_sel, out_sel);

	/**
	 * Remove potential invalid values from selection if the search list
	 * contains default values
	 */
	pvcop::db::array empty_array(column.type(), 1, true);

	PVSelection all_search_values_sel(search_array.size());
	all_search_values_sel.select_all();
	PVSelection default_values_sel(search_array.size());
	default_values_sel.select_none();

	pvcop::db::algo::subselect(search_array, empty_array, all_search_values_sel,
	                           default_values_sel);
	bool remove_invalid_values = not default_values_sel.is_empty();

	if (remove_invalid_values) {
		remove_default_invalid_values(col_idx, out_sel, out_sel);
	}
}

void Inendi::PVLayerFilterMultipleSearch::search_values_if(
    PVCol col,
    const std::vector<std::string>& exps,
    size_t type,
    const PVSelection& in_sel,
    Inendi::PVSelection& out_sel,
    std::function<bool(const std::string&, const std::string&)> predicate) const
{
	PVRush::PVNraw const& nraw = _view->get_rushnraw_parent();

	if (type & VALID) {
		const pvcop::db::array& column = nraw.collection().column(col);
		pvcop::db::algo::subselect_if(column, exps, predicate, in_sel, out_sel);
		remove_default_invalid_values(col, out_sel, out_sel);
	}
	if (type & INVALID) {
		nraw.unconvertable_values_search(col, exps, in_sel, out_sel, predicate);
	}
	if (type & EMPTY) {
		nraw.empty_values_search(col, in_sel, out_sel);
	}
}

PVCore::PVArgumentKeyList Inendi::PVLayerFilterMultipleSearch::get_args_keys_for_preset() const
{
	PVCore::PVArgumentKeyList keys = get_default_args().keys();
	keys.erase(std::find(keys.begin(), keys.end(), ARG_NAME_AXIS));
	return keys;
}

PVCore::PVArgumentList Inendi::PVLayerFilterMultipleSearch::search_value_menu(PVRow /*row*/,
                                                                              PVCombCol /*col*/,
                                                                              PVCol org_col,
                                                                              QString const& v)
{
	PVCore::PVArgumentList args = default_args();

	// Show a carriage return just to be more explicit about the fact we are searching for empty
	// lines
	args[ARG_NAME_EXPS].setValue(PVCore::PVPlainTextType(v.isEmpty() ? "\n" : v));

	args[ARG_NAME_AXIS].setValue(PVCore::PVOriginalAxisIndexType(org_col));

	PVCore::PVEnumType e = args[ARG_NAME_TYPE].value<PVCore::PVEnumType>();
	e.set_sel(2);
	args[ARG_NAME_TYPE].setValue(e);

	args.set_edition_flag(false);

	return args;
}

PVCore::PVArgumentList Inendi::PVLayerFilterMultipleSearch::search_using_value_menu(
    PVRow row, PVCombCol col, PVCol org_col, QString const& v)
{
	PVCore::PVArgumentList args = search_value_menu(row, col, org_col, v);

	// Only search on valid values by default (allows better input checking)
	PVCore::PVEnumType e = args[ARG_NAME_TYPE].value<PVCore::PVEnumType>();
	e.set_sel(0);
	args[ARG_NAME_TYPE].setValue(e);

	args.set_edition_flag(true);

	return args;
}

PVCore::PVArgumentList Inendi::PVLayerFilterMultipleSearch::search_menu(PVRow /*row*/,
                                                                        PVCombCol /*col*/,
                                                                        PVCol org_col,
                                                                        QString const& /*v*/)
{
	PVCore::PVArgumentList args = default_args();

	args[ARG_NAME_AXIS].setValue(PVCore::PVOriginalAxisIndexType(org_col));

	return args;
}

void Inendi::PVLayerFilterMultipleSearch::show_error(QWidget* parent) const
{
	QStringList values;
	for (const std::string& value : _unconverted_values) {
		values << QString::fromStdString(value);
	}

	QMessageBox error_message(
	    QMessageBox::Warning, "Invalid input values",
	    "Some input values failed to be interpreted correctly and were ignored.", QMessageBox::Ok,
	    parent);
	error_message.setDetailedText(values.join("\n"));
	error_message.exec();
}
