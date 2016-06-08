/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <QString>

#include <inendi/PVMandatoryMappingFilter.h>
#include <inendi/PVMapping.h>
#include <inendi/PVMapped.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVSelection.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>

#include <inendi/PVRoot.h>

#include <iostream>

/******************************************************************************
 *
 * Inendi::PVMapped::PVMapped
 *
 *****************************************************************************/
Inendi::PVMapped::PVMapped(PVSource* src)
    : PVCore::PVDataTreeChild<PVSource, PVMapped>(src), _mapping(this)
{
	// FIXME Mapping should be merge in mapped as they are interdependant.
}

/******************************************************************************
 *
 * Inendi::PVMapped::~PVMapped
 *
 *****************************************************************************/
Inendi::PVMapped::~PVMapped()
{
	remove_all_children();
	PVLOG_DEBUG("In PVMapped destructor\n");
}

/******************************************************************************
 *
 * Inendi::PVMapped::allocate_table
 *
 *****************************************************************************/
void Inendi::PVMapped::allocate_table(PVRow const nrows, PVCol const ncols)
{
	_trans_table.resize(ncols);
	for (mapped_row_t& mrow : _trans_table) {
		mrow.resize(nrows);
	}
}

/******************************************************************************
 *
 * Inendi::PVMapped::compute
 *
 *****************************************************************************/
void Inendi::PVMapped::compute()
{
	// Prepare mandatory mapping filters
	std::vector<PVMandatoryMappingFilter::p_type> mand_mapping_filters;
	LIB_CLASS(Inendi::PVMandatoryMappingFilter)
	::list_classes const& lfmf = LIB_CLASS(Inendi::PVMandatoryMappingFilter)::get().get_list();
	mand_mapping_filters.reserve(lfmf.size());
	for (auto it_lfmf = lfmf.begin(); it_lfmf != lfmf.end(); it_lfmf++) {
		PVMandatoryMappingFilter::p_type mf = it_lfmf->value()->clone<PVMandatoryMappingFilter>();
		mand_mapping_filters.push_back(mf);
	}

	const PVRow nrows = get_row_count();

	PVCol const ncols = _mapping.get_number_cols();

	if (nrows == 0) {
		// Nothing to map, early stop.
		return;
	}

	// Prepare the mapping table.
	allocate_table(nrows, ncols);

	// finalize import's mapping filters
	PVRush::PVNraw const& nraw = get_parent()->get_rushnraw();

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
		mapping_filter->init();

		// Compute mapping on this column
		// Set MappingFilter array in filter to set it from filter.
		// FIXME : Ugly interface
		mapping_filter->set_dest_array(nrows, get_column_pointer(j));

		// Set mapping for the full column
		mapping_filter->operator()(j, nraw);

		mandatory_param_map& params_map = _mapping.get_mandatory_params_for_col(j);
// Init the mandatory mapping
// FIXME : This part is critical has filter are function object and they are not
// thread local.
// FIXME : This is a generique handling of mandatory mapping while there is only
// one of these mapping.
// A less generique approach would make the code nicer and thread safe.
#pragma omp critical
		for (auto it_pmf = mand_mapping_filters.begin(); it_pmf != mand_mapping_filters.end();
		     it_pmf++) {
			(*it_pmf)->set_dest_params(params_map);
			(*it_pmf)->set_decimal_type(mapping_filter->get_decimal_type());
			(*it_pmf)->set_mapped(*this);
			(*it_pmf)->operator()(Inendi::mandatory_param_list_values(j, get_column_pointer(j)));
		}

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
 * Inendi::PVMapped::to_csv
 *
 *****************************************************************************/
namespace Inendi
{
namespace __impl
{
struct to_csv_value_holder {
	template <typename T>
	static void
	call(Inendi::PVMapped::mapped_table_t const& trans_table, PVRow const i, PVCol const j)
	{
		std::cout << trans_table[j][i].storage_cast<T>();
	}
};
}
}

void Inendi::PVMapped::to_csv() const
{
	// WARNING: this is all but efficient. Uses this for testing and
	// debugging purpose only !
	for (PVRow i = 0; i < get_row_count(); i++) {
		for (PVCol j = 0; j < get_column_count(); j++) {
			decimal_storage_type::call_from_type<__impl::to_csv_value_holder>(
			    get_decimal_type_of_col(j), _trans_table, i, j);
			if (j != (get_column_count() - 1)) {
				std::cout << ",";
			}
		}
		std::cout << "\n";
	}
}

/******************************************************************************
 *
 * Inendi::PVMapped::get_row_count
 *
 *****************************************************************************/
PVRow Inendi::PVMapped::get_row_count() const
{
	return get_parent<PVSource>()->get_row_count();
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
 * Inendi::PVMapped::process_from_parent_source
 *
 *****************************************************************************/
void Inendi::PVMapped::process_from_parent_source()
{
	compute();
	// Process plotting children
	for (auto* plotted_p : get_children()) {
		plotted_p->process_from_parent_mapped();
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
			PVPlotted_p plotted = emplace_add_child();
			// FIXME : Plotting is created invalid then set
			new_obj->object(QString("plotting"), plotted->get_plotting(), QString(), false, nullptr,
			                false);
			plotted->serialize(*new_obj, so.get_version());
			new_obj->_bound_obj = plotted.get();
			new_obj->_bound_obj_type = typeid(PVPlotted);
			idx++;
		}
	} catch (PVCore::PVSerializeArchiveErrorNoObject const&) {
		return;
	}
}
