/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QList>
#include <QStringList>
#include <QString>

#include <pvkernel/rush/PVFormat.h>

#include <inendi/PVMandatoryMappingFilter.h>
#include <inendi/PVMapping.h>
#include <inendi/PVMapped.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVSelection.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>

#include <inendi/PVRoot.h>

#include <boost/thread.hpp>

#include <iostream>

#include <unordered_set>
#include <float.h>
#include <omp.h>

#include <tbb/parallel_for.h>

/******************************************************************************
 *
 * Inendi::PVMapped::PVMapped
 *
 *****************************************************************************/
Inendi::PVMapped::PVMapped()
{
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

void Inendi::PVMapped::set_parent_from_ptr(PVSource* source)
{
	data_tree_mapped_t::set_parent_from_ptr(source);
	_mapping = PVMapping_p(new PVMapping(this));
}

void Inendi::PVMapped::allocate_table(PVRow const nrows, PVCol const ncols)
{
	_trans_table.resize(ncols);
	for (mapped_row_t& mrow: _trans_table) {
		mrow.resize(nrows);
	}
}

void Inendi::PVMapped::compute()
{
	// Prepare mandatory mapping filters
	std::vector<PVMandatoryMappingFilter::p_type> mand_mapping_filters;
	LIB_CLASS(Inendi::PVMandatoryMappingFilter)::list_classes const& lfmf = LIB_CLASS(Inendi::PVMandatoryMappingFilter)::get().get_list();
	mand_mapping_filters.reserve(lfmf.size());
	for (auto it_lfmf = lfmf.begin(); it_lfmf != lfmf.end(); it_lfmf++) {
		PVMandatoryMappingFilter::p_type mf = it_lfmf->value()->clone<PVMandatoryMappingFilter>();
		mand_mapping_filters.push_back(mf);
	}

	const PVRow nrows = get_parent()->get_row_count();

	PVCol const ncols = _mapping->get_number_cols();

	if (nrows == 0) {
		// Nothing to map, early stop.
		return;
	}

	// Prepare the mapping table.
	allocate_table(nrows, ncols);

	// finalize import's mapping filters
	PVRush::PVNraw const& nraw = get_parent()->get_rushnraw();

#pragma omp parallel for
	for (PVCol j = 0; j < ncols; j++) {
		// Check that an update is required
		if (_mapping->get_properties_for_col(j).is_uptodate()) {
			continue;
		}

		// Create our own plugins from the library
		PVMappingFilter::p_type mf = _mapping->get_filter_for_col(j);
		PVMappingFilter::p_type mapping_filter = mf->clone<PVMappingFilter>();
		mapping_filter->init();

		// Compute mapping on this column
		// Set MappingFilter array in filter to set it from filter.
		// FIXME : Ugly interface
		mapping_filter->set_dest_array(nrows, get_column_pointer(j));
		
		// Set mapping for the full column
		mapping_filter->operator()(j, nraw);

		mandatory_param_map& params_map = _mapping->get_mandatory_params_for_col(j);
		tbb::tick_count tmap_start = tbb::tick_count::now();
		// Init the mandatory mapping
		for (auto it_pmf = mand_mapping_filters.begin(); it_pmf != mand_mapping_filters.end(); it_pmf++) {
			(*it_pmf)->set_dest_params(params_map);
			(*it_pmf)->set_decimal_type(mapping_filter->get_decimal_type());
			(*it_pmf)->set_mapped(*this);
			(*it_pmf)->operator()(Inendi::mandatory_param_list_values(j, get_column_pointer(j)));
		}
		tbb::tick_count tmap_end = tbb::tick_count::now();

		PVLOG_INFO("(PVMapped) mandatory mapping for axis %d took %0.4f seconds.\n", j, (tmap_end-tmap_start).seconds());
		_mapping->set_uptodate_for_col(j);
		invalidate_plotted_children_column(j);
	}

	// force plotteds updates (in case of .pvi load)
	for (auto plotted : get_children<PVPlotted>()) {
		plotted->finish_process_from_rush_pipeline();
	}
}

/******************************************************************************
 *
 * Inendi::PVMapped::to_csv
 *
 *****************************************************************************/
namespace Inendi { namespace __impl {
struct to_csv_value_holder
{
	template <typename T>
	static void call(Inendi::PVMapped::mapped_table_t const& trans_table, PVRow const i, PVCol const j)
	{
		std::cout << trans_table[j][i].storage_cast<T>();
	}
};
} }

void Inendi::PVMapped::to_csv()
{
	// WARNING: this is all but efficient. Uses this for testing and
	// debugging purpose only !
	for (PVRow i = 0; i < get_row_count(); i++) {
		for (PVCol j = 0; j < get_column_count(); j++) {
			decimal_storage_type::call_from_type<__impl::to_csv_value_holder>(get_decimal_type_of_col(j), _trans_table, i, j);
			if (j != (get_column_count()-1)) {
				std::cout << ",";
			}
		}
		std::cout << "\n";
	}
}

/******************************************************************************
 *
 * Inendi::PVMapped::get_format
 *
 *****************************************************************************/
PVRush::PVFormat_p Inendi::PVMapped::get_format() const
{
	return get_parent()->get_rushnraw().get_format();
}

/******************************************************************************
 *
 * Inendi::PVMapped::get_sub_col_minmax
 *
 *****************************************************************************/
namespace Inendi { namespace __impl {
struct get_sub_col_minmax_holder
{
	template <typename T>
	static void call(Inendi::PVMapped::mapped_sub_col_t& ret, Inendi::PVMapped::decimal_storage_type& min, Inendi::PVMapped::decimal_storage_type& max, Inendi::PVSelection const& sel, PVCol const col, Inendi::PVMapped::mapped_table_t const& trans_table)
	{
		min.set_max<T>();
		max.set_min<T>();

		const Inendi::PVMapped::decimal_storage_type* mapped_values = &trans_table[col][0];
		T& max_cast = max.storage_cast<T>();
		T& min_cast = min.storage_cast<T>();
		sel.visit_selected_lines([&](PVRow const i){
			const Inendi::PVMapped::decimal_storage_type v = mapped_values[i];
			const T v_cast = v.storage_cast<T>();
			if (v_cast > max_cast) {
				max_cast = v_cast;
			}
			if (v_cast < min_cast) {
				min_cast = v_cast;
			}
			ret.push_back(Inendi::PVMapped::mapped_sub_col_t::value_type(i, v));
		},
		trans_table[0].size());
	}
};
} }

void Inendi::PVMapped::get_sub_col_minmax(mapped_sub_col_t& ret, decimal_storage_type& min, decimal_storage_type& max, PVSelection const& sel, PVCol const col) const
{
	PVCore::DecimalType const type_col = get_decimal_type_of_col(col);
	PVRow size = get_row_count();
	ret.reserve(sel.get_number_of_selected_lines_in_range(0, size-1));

	decimal_storage_type::call_from_type<__impl::get_sub_col_minmax_holder>(type_col, ret, min, max, sel, col, _trans_table);
}

/******************************************************************************
 *
 * Inendi::PVMapped::get_row_count
 *
 *****************************************************************************/
PVRow Inendi::PVMapped::get_row_count() const
{
	return get_parent()->get_row_count();
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

void Inendi::PVMapped::add_column(PVMappingProperties const& props)
{
	_mapping->add_column(props);
}

void Inendi::PVMapped::process_parent_source()
{
	compute();
}

void Inendi::PVMapped::process_from_parent_source()
{
	process_parent_source();
	// Process plotting children
	for (auto plotted_p : get_children<PVPlotted>()) {
		plotted_p->process_from_parent_mapped();
	}
}

void Inendi::PVMapped::invalidate_plotted_children_column(PVCol j)
{
	for (auto plotted_p : get_children<PVPlotted>()) {
		plotted_p->invalidate_column(j);
	}
}

void Inendi::PVMapped::invalidate_all()
{
	_mapping->invalidate_all();
}

void Inendi::PVMapped::validate_all()
{
	_mapping->validate_all();
}

bool Inendi::PVMapped::is_current_mapped() const
{
	Inendi::PVView const* cur_view = get_parent<PVSource>()->current_view();
	for (auto const& cv: get_children<Inendi::PVView>()) {
		if (cv.get() == cur_view) {
			return true;
		}
	}
	return false;
}

void Inendi::PVMapped::serialize_write(PVCore::PVSerializeObject& so)
{
	data_tree_mapped_t::serialize_write(so);

	so.object(QString("mapping"), *_mapping, QString(), false, (PVMapping*) NULL, false);
}

void Inendi::PVMapped::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v)
{
	PVMapping* mapping = new PVMapping();
	so.object(QString("mapping"), *mapping, QString(), false, (PVMapping*) NULL, false);
	_mapping = PVMapping_p(mapping);
	_mapping->set_mapped(this);

	// It important to deserialize the children after the mapping otherwise PVPlottingProperties complains that there is no mapping!
	data_tree_mapped_t::serialize_read(so, v);
}
