/**
 * \file PVLayerFilterMultipleSearch.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVLayerFilterMultipleSearch.h"
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>
#include <picviz/PVView.h>

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
 * Picviz::PVLayerFilterMultipleSearch::PVLayerFilterMultipleSearch
 *
 *****************************************************************************/
Picviz::PVLayerFilterMultipleSearch::PVLayerFilterMultipleSearch(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterMultipleSearch, l);
	add_ctxt_menu_entry("Search for this value", &PVLayerFilterMultipleSearch::search_value_menu);
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
	args[PVCore::PVArgumentKey(ARG_NAME_AXIS, QObject::tr(ARG_DESC_AXIS))].setValue(PVCore::PVOriginalAxisIndexType(0));
	args[PVCore::PVArgumentKey(ARG_NAME_INCLUDE, QObject::tr(ARG_DESC_INCLUDE))].setValue(PVCore::PVEnumType(QStringList() << QString("include") << QString("exclude"), 0));
	args[PVCore::PVArgumentKey(ARG_NAME_CASE, QObject::tr(ARG_DESC_CASE))].setValue(PVCore::PVEnumType(QStringList() << QString("Does not match case") << QString("Match case") , 0));
	args[PVCore::PVArgumentKey(ARG_NAME_ENTIRE, QObject::tr(ARG_DESC_ENTIRE))].setValue(PVCore::PVEnumType(QStringList() << QString("Part of the field") << QString("The entire field") , 0));
	args[PVCore::PVArgumentKey(ARG_NAME_INTERPRET, QObject::tr(ARG_DESC_INTERPRET))].setValue(PVCore::PVEnumType(QStringList() << QString("Plain text") << QString("Regular expressions"), 0));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterMultipleSearch::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterMultipleSearch::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_id = _args[ARG_NAME_AXIS].value<PVCore::PVOriginalAxisIndexType>().get_original_index();
	int interpret = _args[ARG_NAME_INTERPRET].value<PVCore::PVEnumType>().get_sel_index();
	bool include = _args[ARG_NAME_INCLUDE].value<PVCore::PVEnumType>().get_sel_index() == 0;
	bool case_match = _args[ARG_NAME_CASE].value<PVCore::PVEnumType>().get_sel_index() == 1;
	bool exact_match = _args[ARG_NAME_ENTIRE].value<PVCore::PVEnumType>().get_sel_index() == 1;
	bool is_rx = interpret >= 1;
	//bool is_wildcard = interpret == 2;

	QString const& txt = _args[ARG_NAME_EXPS].value<PVCore::PVPlainTextType>().get_text();
	QStringList exps = txt.split("\n");
	std::vector<QByteArray> exps_utf8;
	std::vector<pcrecpp::RE> rxs;
	if (is_rx) {
		int flags = PCRE_UTF8;
		if (!case_match) {
			flags |= PCRE_CASELESS;
		}
		rxs.reserve(exps.size());
		for (int i = 0; i < exps.size(); i++) {
			QString pattern = exps.at(i).trimmed();
			if (!pattern.isEmpty()) {
				rxs.emplace_back(pattern.toUtf8().constData(), flags);
			}
		}
	}
	else {
		exps_utf8.resize(exps.size());
#pragma omp parallel for
		for (int i = 0; i < exps.size(); i++) {
			QString const& str = exps[i];
			if (!str.isEmpty()) {
				exps_utf8[i] = str.trimmed().toUtf8();
			}
		}
	}

	PVRush::PVNraw const& nraw = _view->get_rushnraw_parent();

	tbb::enumerable_thread_specific<PVSelection> tls_sel;

	char* old_locale = setlocale(LC_COLLATE, NULL);
	setlocale(LC_COLLATE, "fr_FR.UTF-8");
	BENCH_START(visit);
	nraw.visit_column_tbb_sel(axis_id, [&](PVRow const r, const char* buf, size_t n)
		{
			//QString str(QString::fromUtf8(buf, n));
			bool sel = false;
			if (is_rx) {
				for (pcrecpp::RE const& re: rxs) {
					// Local copy of object
					pcrecpp::RE my_re = re;
					bool found;
					if (exact_match) {
						found = my_re.FullMatch(pcrecpp::StringPiece(buf, n));
					}
					else {
						found = my_re.PartialMatch(pcrecpp::StringPiece(buf, n));
					}
					if (found) {
						sel = true;
						break;
					}
				}
			}
			else {
				for (int i = 0; i < exps.size(); i++) {
					QByteArray const& exp = exps_utf8[i];
					if (exp.isEmpty()) {
						continue;
					}
					if (exact_match) {
						bool found;
						if (case_match) {
							found = (PVCore::PVUnicodeString(buf, n) == PVCore::PVUnicodeString(exp.constData(), exp.size()));
						}
						else {
							found = (PVCore::PVUnicodeString(buf, n).compareNoCase(PVCore::PVUnicodeString(exp.constData(), exp.size())) == 0);
						}
						if (found) {
							sel = true;
							break;
						}
					}
					else {
						bool found;
						if (case_match) {
							found = (strstr(buf, exp.constData()) != NULL);
						}
						else {
							found = (strcasestr(buf, exp.constData()) != NULL);
						}
						if (found) {
							sel = true;
							break;
						}
					}
				}
			}

			sel = !(sel ^ include);
			tls_sel.local().set_line(r, sel);
		}, _view->get_pre_filter_layer().get_selection());
	BENCH_END(visit, "multiple-search", 1, 1, 1, 1);

	setlocale(LC_COLLATE, old_locale);

	typename decltype(tls_sel)::const_iterator it_tls = tls_sel.begin();
	PVSelection& out_sel = out.get_selection();
	// Save one copy with std::move :) !
	out_sel = std::move(*it_tls);
	it_tls++;
	for (; it_tls != tls_sel.end(); it_tls++) {
		out_sel.or_optimized(*it_tls);
	}
}

QList<PVCore::PVArgumentKey> Picviz::PVLayerFilterMultipleSearch::get_args_keys_for_preset() const
{
	QList<PVCore::PVArgumentKey> keys = get_default_args().keys();
	keys.removeAll(ARG_NAME_AXIS);
	return keys;
}

PVCore::PVArgumentList Picviz::PVLayerFilterMultipleSearch::search_value_menu(PVRow /*row*/, PVCol /*col*/, PVCol org_col, QString const& v)
{
	PVCore::PVArgumentList args = default_args();
	args[ARG_NAME_EXPS].setValue(PVCore::PVPlainTextType(v));
	args[ARG_NAME_AXIS].setValue(PVCore::PVOriginalAxisIndexType(org_col));

	PVCore::PVEnumType e = args[ARG_NAME_CASE].value<PVCore::PVEnumType>();
	e.set_sel(1);
	args[ARG_NAME_CASE].setValue(e);

	e = args[ARG_NAME_ENTIRE].value<PVCore::PVEnumType>();
	e.set_sel(1);
	args[ARG_NAME_ENTIRE].setValue(e);

	return args;
}

IMPL_FILTER(Picviz::PVLayerFilterMultipleSearch)
