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

#include <inendi/PVMapped.h>            // for PVMapped, etc
#include <inendi/PVMappingFilter.h>     // for PVMappingFilter, etc
#include <inendi/PVMappingProperties.h> // for PVMappingProperties
#include <inendi/PVPlotted.h>           // for PVPlotted
#include <inendi/PVSource.h>            // for PVSource

#include <pvkernel/core/PVDataTreeObject.h>  // for PVDataTreeChild
#include <pvkernel/core/PVLogger.h>          // for PVLOG_DEBUG, PVLOG_ERROR, etc
#include <pvkernel/core/PVSerializeObject.h> // for PVSerializeObject_p, etc

#include <pvbase/types.h> // for PVCol, PVRow

#include <pvkernel/core/inendi_bench.h> // for BENCH_END, BENCH_START

#include <QString> // for QString

#include <algorithm> // for move
#include <cassert>   // for assert
#include <list>      // for list
#include <memory>    // for __shared_ptr
#include <string>    // for string

namespace PVRush
{
class PVNraw;
} // namespace PVRush

/******************************************************************************
 *
 * Inendi::PVMapped::PVMapped
 *
 *****************************************************************************/
Inendi::PVMapped::PVMapped(PVSource& src, std::string const& name)
    : PVCore::PVDataTreeChild<PVSource, PVMapped>(src), _name(name)
{
	// FIXME : Should be const
	PVSource& source = get_parent();

	PVCol naxes = source.get_nraw_column_count();

	if (naxes == 0) {
		PVLOG_ERROR("In PVMapping constructor, no axis have been defined in the "
		            "format !!!!\n");
		assert(false);
	}

	PVLOG_DEBUG("In PVMapping::PVMapping(), debug PVFormat\n");
	for (PVCol i(0); i < naxes; i++) {
		columns.emplace_back(source.get_format(), i);
		PVLOG_HEAVYDEBUG("%s: Add a column\n", __FUNCTION__);
	}

	compute();
}

Inendi::PVMapped::PVMapped(PVSource& src,
                           std::string const& name,
                           std::list<Inendi::PVMappingProperties>&& columns)
    : PVCore::PVDataTreeChild<PVSource, PVMapped>(src), columns(std::move(columns)), _name(name)
{
	compute();
}

/******************************************************************************
 *
 * Inendi::PVMapped::compute
 *
 *****************************************************************************/
void Inendi::PVMapped::compute()
{
	if (get_row_count() == 0) {
		// Nothing to map, early stop.
		return;
	}

	PVCol const ncols(columns.size());

	// Prepare the mapping table.
	_trans_table.resize(ncols);

	// finalize import's mapping filters
	PVRush::PVNraw const& nraw = get_parent().get_rushnraw();

	/**
	 * For now, the mapping parallelization is only done by rows
	 * but when we will want to parallelise the computation of the mapping also by
	 * column
	 * (to speed up the recomputation all the mappings at once) we should
	 * carrefelluly
	 * handle this nested parallelization, using tasks for example.
	 */
	for (PVCol j(0); j < ncols; j++) {
		// Check that an update is required
		if (get_properties_for_col(j).is_uptodate()) {
			continue;
		}

		// Create our own plugins from the library
		PVMappingFilter::p_type mf = get_properties_for_col(j).get_mapping_filter();
		PVMappingFilter::p_type mapping_filter = mf->clone<PVMappingFilter>();

		// Set mapping for the full column
		_trans_table[j] = mapping_filter->operator()(j, nraw);

		get_properties_for_col(j).set_minmax(
		    mapping_filter->get_minmax(get_column(j), nraw.column(j).valid_selection()));

		get_properties_for_col(j).set_uptodate();
		invalidate_plotted_children_column(j);
	}
}

/******************************************************************************
 *
 * Inendi::PVMapped::get_column
 *
 *****************************************************************************/
pvcop::db::array const& Inendi::PVMapped::get_column(PVCol col) const
{
	if (_trans_table[col]) {
		return _trans_table[col];
	} else {
		PVRush::PVNraw const& nraw = get_parent().get_rushnraw();
		return nraw.column(col);
	}
}

/******************************************************************************
 *
 * Inendi::PVMapped::append_mapped
 *
 *****************************************************************************/
void Inendi::PVMapped::append_mapped()
{
	columns.emplace_back(PVMappingProperties("default", PVCore::PVArgumentList()));
}

/******************************************************************************
 *
 * Inendi::PVMapped::delete_mapped
 *
 *****************************************************************************/
void Inendi::PVMapped::delete_mapped(PVCol col)
{
	columns.erase(std::next(columns.begin(), col.value()));
}

/******************************************************************************
 *
 * Inendi::PVMapped::get_row_count
 *
 *****************************************************************************/
PVRow Inendi::PVMapped::get_row_count() const
{
	return get_parent<PVSource>().get_row_count();
}

/******************************************************************************
 *
 * Inendi::PVMapped::get_nraw_column_count
 *
 *****************************************************************************/
PVCol Inendi::PVMapped::get_nraw_column_count() const
{
	return PVCol(_trans_table.size());
}

/******************************************************************************
 *
 * Inendi::PVMapped::update_mapping
 *
 *****************************************************************************/
void Inendi::PVMapped::update_mapping()
{
	compute();
	// Process plotting children
	for (auto* plotted_p : get_children()) {
		plotted_p->update_plotting();
	}
}

/******************************************************************************
 *
 * Inendi::PVMapped::invalidate_plotted_children_column
 *
 *****************************************************************************/
void Inendi::PVMapped::invalidate_plotted_children_column(PVCol j)
{
	for (auto* plotted_p : get_children()) {
		plotted_p->invalidate_column(j);
	}
}

/******************************************************************************
 *
 * Inendi::PVMapped::serialize_write
 *
 *****************************************************************************/
void Inendi::PVMapped::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.set_current_status("Saving mapping...");

	QString name = QString::fromStdString(_name);
	so.attribute_write("name", name);

	so.set_current_status("Saving mapping properties...");
	PVCore::PVSerializeObject_p list_prop = so.create_object("properties");

	int idx = 0;
	for (PVMappingProperties const& prop : columns) {
		PVCore::PVSerializeObject_p new_obj = list_prop->create_object(QString::number(idx++));
		prop.serialize_write(*new_obj);
	}
	so.attribute_write("prop_count", idx);

	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj = so.create_object("plotted");
	idx = 0;
	for (PVPlotted const* plotted : get_children()) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx++));
		plotted->serialize_write(*new_obj);
	}
	so.attribute_write("plotted_count", idx);
}

/******************************************************************************
 *
 * Inendi::PVMapped::serialize_read
 *
 *****************************************************************************/
Inendi::PVMapped& Inendi::PVMapped::serialize_read(PVCore::PVSerializeObject& so,
                                                   Inendi::PVSource& parent)
{
	so.set_current_status("Loading mapping...");
	auto name = so.attribute_read<QString>("name");

	PVCore::PVSerializeObject_p list_prop = so.create_object("properties");

	so.set_current_status("Loading mapping properties...");
	std::list<Inendi::PVMappingProperties> columns;
	int prop_count = so.attribute_read<int>("prop_count");
	for (int idx = 0; idx < prop_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_prop->create_object(QString::number(idx));
		columns.emplace_back(PVMappingProperties::serialize_read(*new_obj));
	}

	PVMapped& mapped = parent.emplace_add_child(name.toStdString(), std::move(columns));

	// Create the list of plotted
	PVCore::PVSerializeObject_p list_obj = so.create_object("plotted");
	int plotted_count = so.attribute_read<int>("plotted_count");
	for (int idx = 0; idx < plotted_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
		PVPlotted::serialize_read(*new_obj, mapped);
	}

	return mapped;
}
