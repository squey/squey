/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVLayer.h>
#include <inendi/PVLinesProperties.h> // for PVLinesProperties
#include <inendi/PVPlotted.h>         // for PVPlotted
#include <inendi/PVSelection.h>       // for PVSelection

#include <pvkernel/core/PVSerializeObject.h> // for PVSerializeObject

#include <pvbase/types.h> // for PVCol, PVRow

#include <memory> // for __shared_ptr, shared_ptr

/******************************************************************************
 *
 * Inendi::PVLayer::PVLayer
 *
 *****************************************************************************/
Inendi::PVLayer::PVLayer(const QString& name_, size_t size)
    : index(0), _locked(false), visible(true), name(name_), selection(size), lines_properties(size)
{
	reset_to_full_and_default_color();
}

/******************************************************************************
 *
 * Inendi::PVLayer::PVLayer
 *
 *****************************************************************************/
Inendi::PVLayer::PVLayer(const QString& name_,
                         const PVSelection& sel_,
                         const PVLinesProperties& lp_)
    : index(0), _locked(false), visible(true), name(name_), selection(sel_), lines_properties(lp_)
{
	name.truncate(INENDI_LAYER_NAME_MAXLEN);
}

/******************************************************************************
 *
 * Inendi::PVLayer::A2B_copy_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Inendi::PVLayer::A2B_copy_restricted_by_selection(PVLayer& b, PVSelection const& selection)
{
	get_lines_properties().A2B_copy_restricted_by_selection(b.get_lines_properties(), selection);
	b.get_selection() &= get_selection();
	b.compute_selectable_count();
}

void Inendi::PVLayer::compute_selectable_count()
{
	// FIXME : We should cache this value instead of computing it explicitly
	selectable_count = selection.bit_count();
}

/******************************************************************************
 *
 * Inendi::PVLayer::reset_to_empty_and_default_color
 *
 *****************************************************************************/
void Inendi::PVLayer::reset_to_empty_and_default_color()
{
	lines_properties.reset_to_default_color();
	selection.select_none();
}

/******************************************************************************
 *
 * Inendi::PVLayer::reset_to_default_color
 *
 *****************************************************************************/
void Inendi::PVLayer::reset_to_default_color()
{
	lines_properties.reset_to_default_color();
}

/******************************************************************************
 *
 * Inendi::PVLayer::reset_to_full_and_default_color
 *
 *****************************************************************************/
void Inendi::PVLayer::reset_to_full_and_default_color()
{
	lines_properties.reset_to_default_color();
	selection.select_all();
}

void Inendi::PVLayer::compute_min_max(PVPlotted const& plotted)
{
	PVCol col_count = plotted.get_nraw_column_count();
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

void Inendi::PVLayer::serialize_write(PVCore::PVSerializeObject& so)
{
	auto sel_obj = so.create_object("selection");
	selection.serialize_write(*sel_obj);

	auto lp_obj = so.create_object("lp");
	lines_properties.serialize_write(*lp_obj);

	so.attribute("name", name);
	so.attribute("visible", visible);
	so.attribute("index", index);
	so.attribute("locked", _locked);
}

Inendi::PVLayer Inendi::PVLayer::serialize_read(PVCore::PVSerializeObject& so)
{
	QString name;
	so.attribute("name", name);

	auto sel_obj = so.create_object("selection");
	Inendi::PVSelection sel(Inendi::PVSelection::serialize_read(*sel_obj));

	auto lp_obj = so.create_object("lp");
	Inendi::PVLinesProperties lines_properties = Inendi::PVLinesProperties::serialize_read(*lp_obj);

	Inendi::PVLayer layer(name, sel, lines_properties);
	bool visible;
	so.attribute("visible", visible);
	layer.set_visible(visible);

	int index;
	so.attribute("index", index);
	layer.set_index(index);

	bool locked;
	so.attribute("locked", locked);
	if (locked) {
		layer.set_lock();
	}
	return layer;
}
