/**
 * \file PVLayer.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <picviz/PVLayer.h>
#include <picviz/PVPlotted.h>

// AG: FIXME: we don't have to incldue PVView here. There is a weird issue w/ forward
// declaration and picviz's shared pointer
#include <picviz/PVView.h>

/******************************************************************************
 *
 * Picviz::PVLayer::PVLayer
 *
 *****************************************************************************/
// Picviz::PVLayer::PVLayer(const QString & name_) :
// 	lines_properties(),
// 	name(name_),
// 	selection()
// {
// 	name.truncate(PICVIZ_LAYER_NAME_MAXLEN);
// }

/******************************************************************************
 *
 * Picviz::PVLayer::PVLayer
 *
 *****************************************************************************/
Picviz::PVLayer::PVLayer(const QString & name_, const PVSelection & sel_, const PVLinesProperties & lp_) :
	lines_properties(lp_),
	name(name_),
	selection(sel_)
{
	name.truncate(PICVIZ_LAYER_NAME_MAXLEN);
	locked = false;
	visible = true;
}

/******************************************************************************
 *
 * Picviz::PVLayer::A2B_copy_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Picviz::PVLayer::A2B_copy_restricted_by_selection_and_nelts(PVLayer &b, PVSelection const& selection, PVRow nelts)
{
	get_lines_properties().A2B_copy_restricted_by_selection_and_nelts(b.get_lines_properties(), selection, nelts);
	b.get_selection() &= get_selection();
}

/******************************************************************************
 *
 * Picviz::PVLayer::reset_to_empty_and_default_color
 *
 *****************************************************************************/
void Picviz::PVLayer::reset_to_empty_and_default_color()
{
	lines_properties.reset_to_default_color();
	selection.select_none();
}

/******************************************************************************
 *
 * Picviz::PVLayer::reset_to_full_and_default_color
 *
 *****************************************************************************/
void Picviz::PVLayer::reset_to_full_and_default_color()
{
	lines_properties.reset_to_default_color();
	selection.select_all();
}

void Picviz::PVLayer::compute_min_max(PVPlotted const& plotted)
{
	PVCol col_count = plotted.get_column_count();
	_row_mins.resize(col_count);
	_row_maxs.resize(col_count);

#pragma omp parallel for
	for (PVCol j = 0; j < col_count; j++) {
		PVRow min,max;
		plotted.get_col_minmax(min, max, selection, j);
		_row_mins[j] = min;
		_row_maxs[j] = max;
	}
}

bool Picviz::PVLayer::get_min_for_col(PVCol col, PVRow& row) const
{
	if (col >= (PVCol) _row_mins.size()) {
		return false;
	}

	row = _row_mins[col];
	return true;
}

bool Picviz::PVLayer::get_max_for_col(PVCol col, PVRow& row) const
{
	if (col >= (PVCol) _row_maxs.size()) {
		return false;
	}

	row = _row_maxs[col];
	return true;
}

void Picviz::PVLayer::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.object("selection", selection);
	so.object("lp", lines_properties);
	so.attribute("name", name);
	so.attribute("visible", visible);
	so.attribute("index", index);
	so.attribute("locked", locked);
}


void Picviz::PVLayer::load_from_file(QString const& path)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchiveZip(path, PVCore::PVSerializeArchive::read, PICVIZ_ARCHIVES_VERSION));
	ar->get_root()->object("layer", *this);
	ar->finish();
#endif
}

void Picviz::PVLayer::save_to_file(QString const& path)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchiveZip(path, PVCore::PVSerializeArchive::write, PICVIZ_ARCHIVES_VERSION));
	ar->get_root()->object("layer", *this);
	ar->finish();
#endif
}
