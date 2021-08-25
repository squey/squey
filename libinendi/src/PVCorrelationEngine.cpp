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

#include <inendi/PVCorrelationEngine.h>

#include <inendi/PVSource.h>
#include <inendi/PVView.h>

#include <pvcop/collection.h>
#include <pvcop/db/algo.h>

/**
 * As currently only one correlation per view is supported,
 * adding another correlation for a view will replace the previous one
 */
void Inendi::PVCorrelationEngine::add(const PVCorrelation& c)
{
	assert(c.view1 != c.view2);

	remove(c.view1);
	_correlations.emplace(c.view1, c);
}

void Inendi::PVCorrelationEngine::remove(const Inendi::PVView* view, bool both_ways /*= false*/)
{
	for (const auto& c : _correlations) {
		if (view == c.second.view1) {
			_correlations.erase(c.first);
			break;
		}
	}

	if (not both_ways) {
		return;
	}

	for (const auto& c : _correlations) {
		if (view == c.second.view2) {
			_correlations.erase(c.first);
			break;
		}
	}
}

bool Inendi::PVCorrelationEngine::exists(const PVCorrelation& c) const
{
	auto corr = _correlations.find(c.view1);

	if (corr != _correlations.end()) {

		PVCol c1 = corr->second.col1;
		Inendi::PVView* v2 = corr->second.view2;
		PVCol c2 = corr->second.col2;
		PVCorrelationType type = corr->second.type;

		return c.col1 == c1 and c.view2 == v2 and c.col2 == c2 and c.type == type;
	}

	return false;
}

bool Inendi::PVCorrelationEngine::exists(const Inendi::PVView* view1, PVCol col1) const
{
	auto correlation = _correlations.find(view1);

	if (correlation == _correlations.end()) {
		return false;
	}

	PVCol c1 = correlation->second.col1;

	return col1 == c1;
}

const Inendi::PVCorrelation*
Inendi::PVCorrelationEngine::correlation(const Inendi::PVView* view) const
{
	auto correlation = _correlations.find(view);

	if (correlation == _correlations.end()) {
		return nullptr;
	}

	return &correlation->second;
}

static void process_values(
	const pvcop::db::array& col1_in,
	const pvcop::db::array& col2_in,
	const Inendi::PVSelection& in_sel,
	Inendi::PVSelection& out_sel
)
{
	pvcop::db::array col1_distinct;
	pvcop::db::algo::distinct(col1_in, col1_distinct, in_sel);

	bool is_string = col1_in.is_string();
	if (not is_string) {
		if (col1_distinct.has_invalid()) {
			pvcop::db::algo::subselect(col2_in, col1_distinct.join(col1_distinct.valid_selection()),
			                           col2_in.valid_selection(), out_sel);
		} else {
			pvcop::db::algo::subselect(col2_in, col1_distinct, col2_in.valid_selection(), out_sel);
		}
	}

	// propagate strings or invalid values
	if ((col1_distinct.has_invalid() and col2_in.has_invalid()) or is_string) {
		Inendi::PVSelection invalid_out_sel(out_sel.count());
		invalid_out_sel.select_none();
		pvcop::db::array string_array;
		if (is_string) {
			string_array = std::move(col1_distinct);
		} else { // invalid values
			string_array = col1_distinct.join(col1_distinct.invalid_selection());
		}

		std::vector<std::string> expr(string_array.size());
		for (size_t i = 0; i < string_array.size(); i++) {
			expr[i] = string_array.at(i);
		}

		pvcop::db::algo::subselect(
		    col2_in, pvcop::db::algo::to_array(col2_in, expr),
		    is_string ? pvcop::core::selection() : col2_in.invalid_selection(), invalid_out_sel);

		out_sel |= invalid_out_sel;
	}
}

static void process_range(
	const pvcop::db::array& col1_in,
	const pvcop::db::array& col2_in,
	const Inendi::PVSelection& in_sel,
	Inendi::PVSelection& out_sel
)
{
	pvcop::db::array minmax = pvcop::db::algo::minmax(col1_in, in_sel);
	pvcop::db::algo::range_select(col2_in, minmax.at(0), minmax.at(1), col2_in.valid_selection(), out_sel);
}

Inendi::PVView* Inendi::PVCorrelationEngine::process(const Inendi::PVView* view1)
{
	auto correlation = _correlations.find(view1);

	if (correlation == _correlations.end()) {
		return nullptr;
	}

	const Inendi::PVSource& src1 = view1->get_parent<Inendi::PVSource>();
	PVCol col1 = correlation->second.col1;
	Inendi::PVView* view2 = correlation->second.view2;
	Inendi::PVSource& src2 = view2->get_parent<Inendi::PVSource>();
	PVCol col2 = correlation->second.col2;

	const pvcop::db::array& col1_in = src1.get_rushnraw().column(col1);
	const pvcop::db::array& col2_in = src2.get_rushnraw().column(col2);

	PVSelection out_sel(col2_in.size());
	out_sel.select_none();

	if (correlation->second.type == PVCorrelationType::VALUES) {
		process_values(col1_in, col2_in, view1->get_real_output_selection(), out_sel);
	}
	else if (correlation->second.type == PVCorrelationType::RANGE) {
		process_range(col1_in, col2_in, view1->get_real_output_selection(), out_sel);
	}

	view2->set_selection_view(out_sel);

	return view2;
}
