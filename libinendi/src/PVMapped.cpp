/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <QString>

#include <inendi/PVMapping.h>
#include <inendi/PVMapped.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVSelection.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>

#include <iostream>

/******************************************************************************
 *
 * Inendi::PVMapped::PVMapped
 *
 *****************************************************************************/
Inendi::PVMapped::PVMapped(PVSource& src)
    : PVCore::PVDataTreeChild<PVSource, PVMapped>(src), _mapping(this)
{
	// FIXME Mapping should be merge in mapped as they are interdependant.
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

	PVCol const ncols = _mapping.get_number_cols();

	// Prepare the mapping table.
	_trans_table.resize(ncols);

	// finalize import's mapping filters
	PVRush::PVNraw const& nraw = get_parent().get_rushnraw();

/**
 * For now, the mapping parallelization is only done by column
 * but when we will want to parallelise the computation of the mapping also by
 * rows
 * (to speed up the recomputation of one specific mapping) we should
 * carrefelluly
 * handle this nested parallelization, using tasks for example.
 */
#pragma omp parallel for
	for (PVCol j = 0; j < ncols; j++) {
		// Check that an update is required
		if (_mapping.get_properties_for_col(j).is_uptodate()) {
			continue;
		}

		// Create our own plugins from the library
		PVMappingFilter::p_type mf = _mapping.get_filter_for_col(j);
		PVMappingFilter::p_type mapping_filter = mf->clone<PVMappingFilter>();

		// Set mapping for the full column
		_trans_table[j] = mapping_filter->operator()(j, nraw);

		_mapping.get_properties_for_col(j).set_minmax(mapping_filter->get_minmax(_trans_table[j]));

		_mapping.set_uptodate_for_col(j);
		invalidate_plotted_children_column(j);
	}

	// force plotteds updates (in case of .pvi load)
	for (auto* plotted : get_children()) {
		plotted->finish_process_from_rush_pipeline();
	}
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
 * Inendi::PVMapped::get_column_count
 *
 *****************************************************************************/
PVCol Inendi::PVMapped::get_column_count() const
{
	return _trans_table.size();
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
 * Inendi::PVMapped::is_current_mapped
 *
 *****************************************************************************/
bool Inendi::PVMapped::is_current_mapped() const
{
	auto children = get_children();
	return std::find_if(children.begin(), children.end(), [](const PVPlotted* plotted) {
		       return plotted->is_current_plotted();
		   }) != children.end();
}

/******************************************************************************
 *
 * Inendi::PVMapped::serialize_write
 *
 *****************************************************************************/
void Inendi::PVMapped::serialize_write(PVCore::PVSerializeObject& so)
{
	so.object(QString("mapping"), _mapping, QString(), false, nullptr, false);

	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);
	int idx = 0;
	for (PVPlotted* plotted : get_children()) {
		QString child_name = QString::number(idx++);
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(
		    child_name, QString::fromStdString(plotted->get_serialize_description()), false);
		plotted->serialize(*new_obj, so.get_version());
		new_obj->_bound_obj = plotted;
		new_obj->_bound_obj_type = typeid(PVPlotted);
	}
}

/******************************************************************************
 *
 * Inendi::PVMapped::serialize_read
 *
 *****************************************************************************/
void Inendi::PVMapped::serialize_read(PVCore::PVSerializeObject& so)
{
	// Create the list of plotted
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);
	int idx = 0;
	try {
		while (true) {
			// FIXME It throws when there are no more data collections.
			// It should not be an exception as it is a normal behavior.
			PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
			PVPlotted& plotted = emplace_add_child();
			// FIXME : Plotting is created invalid then set
			new_obj->object(QString("plotting"), plotted.get_plotting(), QString(), false, nullptr,
			                false);
			plotted.serialize(*new_obj, so.get_version());
			new_obj->_bound_obj = &plotted;
			new_obj->_bound_obj_type = typeid(PVPlotted);
			idx++;
		}
	} catch (PVCore::PVSerializeArchiveErrorNoObject const&) {
		return;
	}
}
