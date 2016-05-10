/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVLayer.h>
#include <inendi/PVPlotted.h>

/******************************************************************************
 *
 * Inendi::PVLayer::PVLayer
 *
 *****************************************************************************/
Inendi::PVLayer::PVLayer(const QString& name_) : index(0), locked(false), visible(true), name(name_)
{
}

/******************************************************************************
 *
 * Inendi::PVLayer::PVLayer
 *
 *****************************************************************************/
Inendi::PVLayer::PVLayer(const QString& name_,
                         const PVSelection& sel_,
                         const PVLinesProperties& lp_)
    : index(0), locked(false), visible(true), name(name_), selection(sel_), lines_properties(lp_)
{
	set_count(selection.count());
	name.truncate(INENDI_LAYER_NAME_MAXLEN);
}

/******************************************************************************
 *
 * Inendi::PVLayer::A2B_copy_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Inendi::PVLayer::A2B_copy_restricted_by_selection_and_nelts(PVLayer& b,
                                                                 PVSelection const& selection,
                                                                 PVRow nelts)
{
	get_lines_properties().A2B_copy_restricted_by_selection_and_nelts(b.get_lines_properties(),
	                                                                  selection, nelts);
	b.get_selection() &= get_selection();
	b.compute_selectable_count(nelts);
}

void Inendi::PVLayer::compute_selectable_count(PVRow const& nrows)
{
	selectable_count = selection.get_number_of_selected_lines_in_range(0U, nrows);
}

/******************************************************************************
 *
 * Inendi::PVLayer::reset_to_empty_and_default_color
 *
 *****************************************************************************/
void Inendi::PVLayer::reset_to_empty_and_default_color(PVRow row_count)
{
	lines_properties.reset_to_default_color(row_count);
	selection.select_none();
}

/******************************************************************************
 *
 * Inendi::PVLayer::reset_to_default_color
 *
 *****************************************************************************/
void Inendi::PVLayer::reset_to_default_color(PVRow row_count)
{
	lines_properties.reset_to_default_color(row_count);
}

/******************************************************************************
 *
 * Inendi::PVLayer::reset_to_full_and_default_color
 *
 *****************************************************************************/
void Inendi::PVLayer::reset_to_full_and_default_color(PVRow row_count)
{
	lines_properties.reset_to_default_color(row_count);
	selection.select_all();
}

void Inendi::PVLayer::compute_min_max(PVPlotted const& plotted)
{
	PVCol col_count = plotted.get_column_count();
	_row_mins.resize(col_count);
	_row_maxs.resize(col_count);

#pragma omp parallel for
	for (PVCol j = 0; j < col_count; j++) {
		PVRow min, max;
		plotted.get_col_minmax(min, max, selection, j);
		_row_mins[j] = min;
		_row_maxs[j] = max;
	}
}

bool Inendi::PVLayer::get_min_for_col(PVCol col, PVRow& row) const
{
	if (col >= (PVCol)_row_mins.size()) {
		return false;
	}

	row = _row_mins[col];
	return true;
}

bool Inendi::PVLayer::get_max_for_col(PVCol col, PVRow& row) const
{
	if (col >= (PVCol)_row_maxs.size()) {
		return false;
	}

	row = _row_maxs[col];
	return true;
}

void Inendi::PVLayer::serialize(PVCore::PVSerializeObject& so,
                                PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.object("selection", selection, "selection", true, (PVSelection*)NULL, false);
	so.object("lp", lines_properties, "lp", true, (PVLinesProperties*)NULL, false);
	so.attribute("name", name);
	so.attribute("visible", visible);
	so.attribute("index", index);
	so.attribute("locked", locked);
}
