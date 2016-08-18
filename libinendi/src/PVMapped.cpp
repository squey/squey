/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <QString>

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
Inendi::PVMapped::PVMapped(PVSource& src, std::string const& name)
    : PVCore::PVDataTreeChild<PVSource, PVMapped>(src), _name(name)
{
	// FIXME : Should be const
	PVSource& source = get_parent();

	PVCol naxes = source.get_column_count();

	if (naxes == 0) {
		PVLOG_ERROR("In PVMapping constructor, no axis have been defined in the "
		            "format !!!!\n");
		assert(false);
	}

	PVLOG_DEBUG("In PVMapping::PVMapping(), debug PVFormat\n");
	for (PVCol i = 0; i < naxes; i++) {
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

	PVCol const ncols = columns.size();

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
		if (get_properties_for_col(j).is_uptodate()) {
			continue;
		}

		// Create our own plugins from the library
		PVMappingFilter::p_type mf = get_properties_for_col(j).get_mapping_filter();
		PVMappingFilter::p_type mapping_filter = mf->clone<PVMappingFilter>();

		// Set mapping for the full column
		_trans_table[j] = mapping_filter->operator()(j, nraw);

		get_properties_for_col(j).set_minmax(mapping_filter->get_minmax(_trans_table[j]));

		get_properties_for_col(j).set_uptodate();
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
 * Inendi::PVMapped::serialize_write
 *
 *****************************************************************************/
void Inendi::PVMapped::serialize_write(PVCore::PVSerializeObject& so)
{

	QString name = QString::fromStdString(_name);
	so.attribute("name", name);

	PVCore::PVSerializeObject_p list_prop = so.create_object("properties", "", true, true);

	int idx = 0;
	for (PVMappingProperties& prop : columns) {
		QString child_name = QString::number(idx++);
		PVCore::PVSerializeObject_p new_obj = list_prop->create_object(child_name, "", false);
		prop.serialize_write(*new_obj);
		new_obj->set_bound_obj(prop);
	}

	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);
	idx = 0;
	for (PVPlotted* plotted : get_children()) {
		QString child_name = QString::number(idx++);
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(
		    child_name, QString::fromStdString(plotted->get_serialize_description()), false);
		plotted->serialize_write(*new_obj);
		new_obj->set_bound_obj(*plotted);
	}
}

/******************************************************************************
 *
 * Inendi::PVMapped::serialize_read
 *
 *****************************************************************************/
Inendi::PVMapped& Inendi::PVMapped::serialize_read(PVCore::PVSerializeObject& so,
                                                   Inendi::PVSource& parent)
{
	QString name;
	so.attribute("name", name);

	PVCore::PVSerializeObject_p list_prop = so.create_object("properties", "", true, true);

	int idx = 0;
	std::list<Inendi::PVMappingProperties> columns;
	try {
		while (true) {
			PVCore::PVSerializeObject_p new_obj = list_prop->create_object(QString::number(idx++));
			columns.emplace_back(PVMappingProperties::serialize_read(*new_obj));
		}
	} catch (PVCore::PVSerializeArchiveErrorNoObject const& /*e*/) {
	}

	PVMapped& mapped = parent.emplace_add_child(name.toStdString(), std::move(columns));

	// Create the list of plotted
	PVCore::PVSerializeObject_p list_obj = so.create_object(
	    mapped.get_children_serialize_name(), mapped.get_children_description(), true, true);
	idx = 0;
	try {
		while (true) {
			// FIXME It throws when there are no more data collections.
			// It should not be an exception as it is a normal behavior.
			PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx++));
			PVPlotted::serialize_read(*new_obj, mapped);
		}
	} catch (PVCore::PVSerializeArchiveErrorNoObject const&) {
	}
	return mapped;
}
