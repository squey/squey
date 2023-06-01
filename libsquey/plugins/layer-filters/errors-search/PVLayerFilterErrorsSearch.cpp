//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVLayerFilterErrorsSearch.h"

#include <squey/PVView.h>

#include <pvkernel/rush/PVNraw.h>

#include <pvkernel/core/squey_bench.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>

#include <pvcop/db/algo.h>

#include <QMessageBox>

#define ARG_NAME_AXIS "axis"
#define ARG_DESC_AXIS "Axis"
#define ARG_NAME_TYPE "type"
#define ARG_DESC_TYPE "Search for"
#define ARG_NAME_INCLUDE "include"
#define ARG_DESC_INCLUDE "Include or exclude pattern"

/******************************************************************************
 *
 * Squey::PVLayerFilterErrorsSearch::PVLayerFilterErrorsSearch
 *
 *****************************************************************************/
Squey::PVLayerFilterErrorsSearch::PVLayerFilterErrorsSearch(PVCore::PVArgumentList const& l)
    : PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterErrorsSearch, l);
	add_ctxt_menu_entry("Search for special values", &PVLayerFilterErrorsSearch::menu);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Squey::PVLayerFilterErrorsSearch)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Squey::PVLayerFilterErrorsSearch)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey(ARG_NAME_AXIS, QObject::tr(ARG_DESC_AXIS))].setValue(
	    PVCore::PVOriginalAxisIndexType(PVCol(0)));
	args[PVCore::PVArgumentKey(ARG_NAME_TYPE, QObject::tr(ARG_DESC_TYPE))].setValue(
	    PVCore::PVEnumType(QStringList() << QString("Empty values") << QString("Invalid values")
	                                     << QString("Empty and invalid values"),
	                       2));
	args[PVCore::PVArgumentKey(ARG_NAME_INCLUDE, QObject::tr(ARG_DESC_INCLUDE))].setValue(
	    PVCore::PVEnumType(QStringList() << QString("include") << QString("exclude"), 0));
	return args;
}

enum ESearchOptions { EMPTY = 1, INVALID = 2 };

/******************************************************************************
 *
 * Squey::PVLayerFilterErrorsSearch::operator()
 *
 *****************************************************************************/

void Squey::PVLayerFilterErrorsSearch::operator()(PVLayer const& in, PVLayer& out)
{
	BENCH_START(errors_search);

	PVCol col(_args[ARG_NAME_AXIS].value<PVCore::PVOriginalAxisIndexType>().get_original_index());
	bool include = _args[ARG_NAME_INCLUDE].value<PVCore::PVEnumType>().get_sel_index() == 0;
	size_t type = _args[ARG_NAME_TYPE].value<PVCore::PVEnumType>().get_sel_index() + 1;

	PVRush::PVNraw const& nraw = _view->get_rushnraw_parent();
	const pvcop::db::array& column = nraw.column(col);

	PVSelection& out_sel = out.get_selection();

	out_sel.select_none();

	if (((type & EMPTY) and column.is_string()) or (type == EMPTY) or
	    (type == INVALID and not column.is_string())) {
		pvcop::db::array empty_array = column.to_array(std::vector<std::string>{{""}});
		pvcop::db::algo::subselect(column, empty_array, in.get_selection(), out_sel);
	}
	if ((type & INVALID) and (not column.is_string())) {
		if (column.invalid_selection()) {
			out_sel = PVSelection((in.get_selection() & column.invalid_selection()) & ~out_sel);
		}
	}

	if (not include) {
		// invert selection
		out_sel = ~out_sel;
		out_sel &= in.get_selection();
	}

	BENCH_END(errors_search, "errors_search", 1, 1, 1, 1);
}

PVCore::PVArgumentKeyList Squey::PVLayerFilterErrorsSearch::get_args_keys_for_preset() const
{
	PVCore::PVArgumentKeyList keys = get_default_args().keys();
	return keys;
}

PVCore::PVArgumentList Squey::PVLayerFilterErrorsSearch::menu(PVRow /*row*/,
                                                               PVCombCol /*col*/,
                                                               PVCol org_col,
                                                               QString const& /*v*/)
{
	PVCore::PVArgumentList args = default_args();

	args[ARG_NAME_AXIS].setValue(PVCore::PVOriginalAxisIndexType(org_col));

	return args;
}
