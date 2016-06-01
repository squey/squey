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

void Inendi::PVCorrelationEngine::add(const Inendi::PVView* view1,
                                      PVCol axis1,
                                      Inendi::PVView* view2,
                                      PVCol axis2)
{
	assert(view1 != view2);

	_correlations.emplace(view1, PVCorrelation{view1, axis1, view2, axis2});
}

void Inendi::PVCorrelationEngine::remove(const Inendi::PVView* view)
{
	_correlations.erase(view);

	for (const auto& correlation : _correlations) {
		const Inendi::PVView* view1 = correlation.first;
		const Inendi::PVView* view2 = correlation.second.view2;

		if (view2 == view) {
			_correlations.erase(view1);
		}
	}
}

bool Inendi::PVCorrelationEngine::exists(const Inendi::PVView* view1,
                                         PVCol col1,
                                         Inendi::PVView* view2,
                                         PVCol col2) const
{
	auto correlation = _correlations.find(view1);

	if (correlation == _correlations.end()) {
		return false;
	}

	PVCol c1 = correlation->second.col1;
	Inendi::PVView* v2 = correlation->second.view2;
	PVCol c2 = correlation->second.col2;

	return col1 == c1 and view2 == v2 and col2 == c2;
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

	const Inendi::PVSource* src1 = view1->get_parent<Inendi::PVSource>();
	PVCol col1 = correlation->second.col1;
	Inendi::PVView* view2 = correlation->second.view2;
	Inendi::PVSource* src2 = view2->get_parent<Inendi::PVSource>();
	PVCol col2 = correlation->second.col2;

	const pvcop::db::array col1_in = src1->get_rushnraw().collection().column(col1);
	const pvcop::db::array col2_in = src2->get_rushnraw().collection().column(col2);

	pvcop::db::array col1_out1;
	pvcop::db::array col1_out2;

	PVSelection s(col2_in.size());
	s.select_all();

	pvcop::db::algo::distinct(col1_in, col1_out1, col1_out2,
	                          view1->get_selection_visible_listing());

	pvcop::db::algo::subselect(col2_in, col1_out1, s,
	                           view2->get_post_filter_layer().get_selection());

	view2->set_selection_view(view2->get_post_filter_layer().get_selection());

	return view2;
}
