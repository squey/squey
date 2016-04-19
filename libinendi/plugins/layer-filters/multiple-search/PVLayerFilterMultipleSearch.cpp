/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVLayerFilterMultipleSearch.h"
#include <pvkernel/core/inendi_bench.h>
#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>
#include <inendi/PVView.h>

#include <pvcop/db/algo.h>
#include <pvcop/core/algo/selection.h>

#include <locale.h>

#include <tbb/enumerable_thread_specific.h>

#include <pcrecpp.h>

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
 * Inendi::PVLayerFilterMultipleSearch::PVLayerFilterMultipleSearch
 *
 *****************************************************************************/
Inendi::PVLayerFilterMultipleSearch::PVLayerFilterMultipleSearch(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterMultipleSearch, l);
	add_ctxt_menu_entry("Search for this value", &PVLayerFilterMultipleSearch::search_value_menu);
	add_ctxt_menu_entry("Search using this value...", &PVLayerFilterMultipleSearch::search_using_value_menu);
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
	args[PVCore::PVArgumentKey(ARG_NAME_EXPS, QObject::tr(ARG_DESC_EXPS))].setValue(PVCore::PVPlainTextType());
	args[PVCore::PVArgumentKey(ARG_NAME_AXIS, QObject::tr(ARG_DESC_AXIS))].setValue(PVCore::PVOriginalAxisIndexType(0));
	args[PVCore::PVArgumentKey(ARG_NAME_INCLUDE, QObject::tr(ARG_DESC_INCLUDE))].setValue(PVCore::PVEnumType(QStringList() << QString("include") << QString("exclude"), 0));
	args[PVCore::PVArgumentKey(ARG_NAME_CASE, QObject::tr(ARG_DESC_CASE))].setValue(PVCore::PVEnumType(QStringList() << QString("Does not match case") << QString("Match case") , 0));
	args[PVCore::PVArgumentKey(ARG_NAME_ENTIRE, QObject::tr(ARG_DESC_ENTIRE))].setValue(PVCore::PVEnumType(QStringList() << QString("Part of the field") << QString("The entire field") , 0));
	args[PVCore::PVArgumentKey(ARG_NAME_INTERPRET, QObject::tr(ARG_DESC_INTERPRET))].setValue(PVCore::PVEnumType(QStringList() << QString("Plain text") << QString("Regular expressions"), 0));
	return args;
}

/******************************************************************************
 *
 * Inendi::PVLayerFilterMultipleSearch::operator()
 *
 *****************************************************************************/
enum ESearchOptions
{
	NONE               = 0,
	REGULAR_EXPRESSION = 1 << 0,
	EXACT_MATCH        = 1 << 1,
	CASE_INSENSITIVE   = 1 << 2
};

void Inendi::PVLayerFilterMultipleSearch::operator()(PVLayer const& in, PVLayer &out)
{
	int axis_id = _args[ARG_NAME_AXIS].value<PVCore::PVOriginalAxisIndexType>().get_original_index();
	int interpret = _args[ARG_NAME_INTERPRET].value<PVCore::PVEnumType>().get_sel_index();
	bool include = _args[ARG_NAME_INCLUDE].value<PVCore::PVEnumType>().get_sel_index() == 0;
	bool case_match = _args[ARG_NAME_CASE].value<PVCore::PVEnumType>().get_sel_index() == 1;
	bool exact_match = _args[ARG_NAME_ENTIRE].value<PVCore::PVEnumType>().get_sel_index() == 1;
	bool is_rx = interpret >= 1;
	//bool is_wildcard = interpret == 2;

	size_t opts = ((REGULAR_EXPRESSION * is_rx) | (EXACT_MATCH * exact_match) | (CASE_INSENSITIVE * (not case_match)));

	const QString& txt = _args[ARG_NAME_EXPS].value<PVCore::PVPlainTextType>().get_text();

	// Remove last carriage return if present otherwise we would search for empty strings as well
	QStringList exps = (txt.right(1) == "\n" ? txt.left(txt.size()-1) : txt).split("\n");

	std::vector<std::string> exps_utf8;
	exps_utf8.resize(exps.size());

#pragma omp parallel for
	for (int i = 0; i < exps.size(); i++) {
		QString const& str = exps[i];
		if (!str.isEmpty()) {
			exps_utf8[i] = str.toUtf8().constData();
		}
	}

	PVRush::PVNraw const& nraw = _view->get_rushnraw_parent();
	PVSelection& out_sel = out.get_selection();

	const pvcop::db::array& column = nraw.collection().column(axis_id);

	BENCH_START(subselect);

	switch(opts) {
		case (EXACT_MATCH) : {
			try {
				pvcop::db::algo::subselect(column, exps_utf8, in.get_selection(), out_sel);
			} catch (pvcop::db::exception::partially_converted_error& e) {
				PVLOG_ERROR("multiple-search : Unable to convert some values");
			}
		}
		break;
		case (EXACT_MATCH | CASE_INSENSITIVE) : {
			pvcop::db::algo::subselect_if(column, exps_utf8, [](const std::string& array_value, const std::string& exp_value) {
				return strcasecmp(array_value.c_str(), exp_value.c_str()) == 0;
			}, in.get_selection(), out_sel);
		}
		break;
		case (EXACT_MATCH | REGULAR_EXPRESSION) : {
			pvcop::db::algo::subselect_if(column, exps_utf8, [](const std::string& array_value, const std::string& exp_value) {
				pcrecpp::RE re(exp_value.c_str(), PCRE_UTF8);
				return re.FullMatch(pcrecpp::StringPiece(array_value.c_str(), array_value.size()));
			}, in.get_selection(), out_sel);
		}
		break;
		case (EXACT_MATCH | REGULAR_EXPRESSION | CASE_INSENSITIVE) : {
			pvcop::db::algo::subselect_if(column, exps_utf8, [](const std::string& array_value, const std::string& exp_value) {
				pcrecpp::RE re(exp_value.c_str(), PCRE_UTF8 | PCRE_CASELESS);
				return re.FullMatch(pcrecpp::StringPiece(array_value.c_str(), array_value.size()));
			}, in.get_selection(), out_sel);
		}
		break;
		case (REGULAR_EXPRESSION) : {
			pvcop::db::algo::subselect_if(column, exps_utf8, [](const std::string& array_value, const std::string& exp_value) {
				pcrecpp::RE re(exp_value.c_str(), PCRE_UTF8);
				return re.PartialMatch(pcrecpp::StringPiece(array_value.c_str(), array_value.size()));
			}, in.get_selection(), out_sel);
		}
		break;
		case (REGULAR_EXPRESSION | CASE_INSENSITIVE) : {
			pvcop::db::algo::subselect_if(column, exps_utf8, [](const std::string& array_value, const std::string& exp_value) {
				pcrecpp::RE re(exp_value.c_str(), PCRE_UTF8 | PCRE_CASELESS);
				return re.PartialMatch(pcrecpp::StringPiece(array_value.c_str(), array_value.size()));
			}, in.get_selection(), out_sel);
		}
		break;
		case (CASE_INSENSITIVE) : {
			pvcop::db::algo::subselect_if(column, exps_utf8, [](const std::string& array_value, const std::string& exp_value) {
				return strcasestr(array_value.c_str(), exp_value.c_str()) != nullptr;
			}, in.get_selection(), out_sel);
		}
		break;
		case (NONE) : {
			pvcop::db::algo::subselect_if(column, exps_utf8, [](const std::string& array_value, const std::string& exp_value) {
				return strstr(array_value.c_str(), exp_value.c_str()) != nullptr;
			}, in.get_selection(), out_sel);
		}
		break;
		default : {
			assert(false);
		}
		break;
	}

	BENCH_END(subselect, "subselect", 1, 1, 1, 1);

	if (not include) {
		// invert selection
		BENCH_START(invert_selection);
		pvcop::core::algo::invert_selection(out_sel);
		BENCH_END(invert_selection, "invert_selection", 1, 1, 1, 1);
	}
}

PVCore::PVArgumentKeyList Inendi::PVLayerFilterMultipleSearch::get_args_keys_for_preset() const
{
	PVCore::PVArgumentKeyList keys = get_default_args().keys();
	keys.erase(std::find(keys.begin(), keys.end(), ARG_NAME_AXIS));
	return keys;
}

PVCore::PVArgumentList Inendi::PVLayerFilterMultipleSearch::search_value_menu(PVRow /*row*/, PVCol /*col*/, PVCol org_col, QString const& v)
{
	PVCore::PVArgumentList args = default_args();

	// Show a carriage return just to be more explicit about the fact we are searching for empty lines
	args[ARG_NAME_EXPS].setValue(PVCore::PVPlainTextType(v.isEmpty() ? "\n" : v));

	args[ARG_NAME_AXIS].setValue(PVCore::PVOriginalAxisIndexType(org_col));

	PVCore::PVEnumType e;
	e = args[ARG_NAME_CASE].value<PVCore::PVEnumType>();
	e.set_sel(1);
	args[ARG_NAME_CASE].setValue(e);

	e = args[ARG_NAME_ENTIRE].value<PVCore::PVEnumType>();
	e.set_sel(1);
	args[ARG_NAME_ENTIRE].setValue(e);

	args.set_edition_flag(false);

	return args;
}

PVCore::PVArgumentList Inendi::PVLayerFilterMultipleSearch::search_using_value_menu(PVRow row, PVCol col, PVCol org_col, QString const& v)
{
	PVCore::PVArgumentList args = search_value_menu(row, col, org_col, v);

	args.set_edition_flag(true);

	return args;
}

PVCore::PVArgumentList Inendi::PVLayerFilterMultipleSearch::search_menu(PVRow /*row*/, PVCol /*col*/, PVCol org_col, QString const& /*v*/)
{
	PVCore::PVArgumentList args = default_args();

	args[ARG_NAME_AXIS].setValue(PVCore::PVOriginalAxisIndexType(org_col));

	PVCore::PVEnumType e = args[ARG_NAME_CASE].value<PVCore::PVEnumType>();
	e.set_sel(1);
	args[ARG_NAME_CASE].setValue(e);

	e = args[ARG_NAME_ENTIRE].value<PVCore::PVEnumType>();
	e.set_sel(1);
	args[ARG_NAME_ENTIRE].setValue(e);

	return args;
}

IMPL_FILTER(Inendi::PVLayerFilterMultipleSearch)
