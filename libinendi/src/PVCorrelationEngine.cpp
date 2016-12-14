/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

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

		return c.col1 == c1 and c.view2 == v2 and c.col2 == c2;
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

	pvcop::db::array col1_out1;
	pvcop::db::array col1_out2;

	PVSelection in_sel(col2_in.size());
	in_sel.select_all();

	PVSelection out_sel(in_sel.count());
	out_sel.select_none();

	pvcop::db::algo::distinct(col1_in, col1_out1, col1_out2, view1->get_real_output_selection());

	pvcop::db::algo::subselect(col2_in, col1_out1, in_sel, out_sel);

	view2->set_selection_view(out_sel);

	return view2;
}
