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

#include <squey/PVLayer.h>
#include <squey/PVLinesProperties.h> // for PVLinesProperties
#include <squey/PVScaled.h>         // for PVScaled
#include <squey/PVSelection.h>       // for PVSelection

#include <pvkernel/core/PVSerializeObject.h> // for PVSerializeObject

#include <pvbase/types.h> // for PVCol, PVRow

#include <memory> // for __shared_ptr, shared_ptr

/******************************************************************************
 *
 * Squey::PVLayer::PVLayer
 *
 *****************************************************************************/
Squey::PVLayer::PVLayer(const QString& name_, size_t size)
    : index(0), _locked(false), visible(true), name(name_), selection(size), lines_properties(size)
{
	reset_to_full_and_default_color();
}

/******************************************************************************
 *
 * Squey::PVLayer::PVLayer
 *
 *****************************************************************************/
Squey::PVLayer::PVLayer(const QString& name_,
                         const PVSelection& sel_,
                         const PVLinesProperties& lp_)
    : index(0), _locked(false), visible(true), name(name_), selection(sel_), lines_properties(lp_)
{
	name.truncate(SQUEY_LAYER_NAME_MAXLEN);
}

/******************************************************************************
 *
 * Squey::PVLayer::A2B_copy_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Squey::PVLayer::A2B_copy_restricted_by_selection(PVLayer& b, PVSelection const& selection)
{
	get_lines_properties().A2B_copy_restricted_by_selection(b.get_lines_properties(), selection);
	b.get_selection() &= get_selection();
	b.compute_selectable_count();
}

void Squey::PVLayer::compute_selectable_count()
{
	// FIXME : We should cache this value instead of computing it explicitly
	selectable_count = selection.bit_count();
}

/******************************************************************************
 *
 * Squey::PVLayer::reset_to_empty_and_default_color
 *
 *****************************************************************************/
void Squey::PVLayer::reset_to_empty_and_default_color()
{
	lines_properties.reset_to_default_color();
	selection.select_none();
}

/******************************************************************************
 *
 * Squey::PVLayer::reset_to_default_color
 *
 *****************************************************************************/
void Squey::PVLayer::reset_to_default_color()
{
	lines_properties.reset_to_default_color();
}

/******************************************************************************
 *
 * Squey::PVLayer::reset_to_full_and_default_color
 *
 *****************************************************************************/
void Squey::PVLayer::reset_to_full_and_default_color()
{
	lines_properties.reset_to_default_color();
	selection.select_all();
}

void Squey::PVLayer::compute_min_max(PVScaled const& scaled)
{
	PVCol col_count = scaled.get_nraw_column_count();
	_row_mins.resize(col_count);
	_row_maxs.resize(col_count);

#pragma omp parallel for
	for (PVCol::value_type j = 0; j < col_count; j++) {
		PVRow min, max;
		scaled.get_col_minmax(min, max, selection, PVCol(j));
		_row_mins[j] = min;
		_row_maxs[j] = max;
	}
}

void Squey::PVLayer::serialize_write(PVCore::PVSerializeObject& so) const
{
	auto sel_obj = so.create_object("selection");
	selection.serialize_write(*sel_obj);

	auto lp_obj = so.create_object("lp");
	lines_properties.serialize_write(*lp_obj);

	so.attribute_write("name", name);
	so.attribute_write("visible", visible);
	so.attribute_write("index", index);
	so.attribute_write("locked", _locked);
}

Squey::PVLayer Squey::PVLayer::serialize_read(PVCore::PVSerializeObject& so)
{
	auto name = so.attribute_read<QString>("name");

	auto sel_obj = so.create_object("selection");
	Squey::PVSelection sel(Squey::PVSelection::serialize_read(*sel_obj));

	auto lp_obj = so.create_object("lp");
	Squey::PVLinesProperties lines_properties = Squey::PVLinesProperties::serialize_read(*lp_obj);

	Squey::PVLayer layer(name, sel, lines_properties);
	layer.set_visible(so.attribute_read<bool>("visible"));
	layer.set_index(so.attribute_read<int>("index"));

	if (so.attribute_read<bool>("locked")) {
		layer.set_lock();
	}
	return layer;
}
