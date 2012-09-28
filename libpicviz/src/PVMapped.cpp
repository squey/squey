/**
 * \file PVMapped.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QList>
#include <QStringList>
#include <QString>

#include <pvkernel/rush/PVFormat.h>

#include <picviz/PVMandatoryMappingFilter.h>
#include <picviz/PVMapping.h>
#include <picviz/PVMapped.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVSelection.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include <picviz/PVRoot.h>

#include <boost/thread.hpp>

#include <iostream>

#include <float.h>
#include <omp.h>

#define DEFAULT_MAPPING_NROWS (16*1024*1024)

/******************************************************************************
 *
 * Picviz::PVMapped::PVMapped
 *
 *****************************************************************************/
Picviz::PVMapped::PVMapped()
{
}

/******************************************************************************
 *
 * Picviz::PVMapped::~PVMapped
 *
 *****************************************************************************/
Picviz::PVMapped::~PVMapped()
{
	PVLOG_INFO("In PVMapped destructor\n");
}

void Picviz::PVMapped::set_parent_from_ptr(PVSource* source)
{
	data_tree_mapped_t::set_parent_from_ptr(source);
	_mapping = PVMapping_p(new PVMapping(this));
}

void Picviz::PVMapped::allocate_table(PVRow const nrows, PVCol const ncols)
{
	_trans_table.resize(ncols);
	reallocate_table(nrows);
}

void Picviz::PVMapped::reallocate_table(PVRow const nrows)
{
	for (mapped_row_t& mrow: _trans_table) {
		mrow.resize(nrows);
	}
}

void Picviz::PVMapped::init_process_from_rush_pipeline()
{
	PVCol const ncols = _mapping->get_number_cols();
	// Let's prepare the mappedtable for 10 million rows, and grow every 10 millions elements.
	// No matter if we go too far, this is just virtual memory, and isn't physically allocated
	// if nothing is written in it !
	allocate_table(DEFAULT_MAPPING_NROWS, ncols);

	// Create our own plugins from the library
	_mapping_filters_rush.resize(ncols);
	for (PVCol j = 0; j < ncols; j++) {
		PVMappingFilter::p_type mf = _mapping->get_filter_for_col(j);
		if (mf) {
			_mapping_filters_rush[j] = mf->clone<PVMappingFilter>();
			_mapping_filters_rush[j]->init();
		}
	}

	// Clear "group values" hash (just in case..)
	_grp_values_rush.clear();
}

void Picviz::PVMapped::init_pure_mapping_functions(PVFilter::PVPureMappingProcessing::list_pure_mapping_t& funcs)
{
	const PVCol ncols = _mapping->get_number_cols();
	assert((PVCol) _mapping_filters_rush.size() == ncols);
	funcs.clear();
	funcs.resize(ncols);
	for (PVCol c = 0; c < ncols; c++) {
		if (is_mapping_pure(c)) {
			funcs[c] = _mapping_filters_rush[c]->f();
		}
	}
}

void Picviz::PVMapped::finish_process_from_rush_pipeline()
{
	// Give back unused memory (over-allocated)
	reallocate_table(get_parent()->get_row_count());

	// Process mandatory mapping filters
	std::vector<PVMandatoryMappingFilter::p_type> mand_mapping_filters;
	LIB_CLASS(Picviz::PVMandatoryMappingFilter)::list_classes const& lfmf = LIB_CLASS(Picviz::PVMandatoryMappingFilter)::get().get_list();
	LIB_CLASS(Picviz::PVMandatoryMappingFilter)::list_classes::const_iterator it_lfmf;
	mand_mapping_filters.reserve(lfmf.size());
	for (it_lfmf = lfmf.begin(); it_lfmf != lfmf.end(); it_lfmf++) {
		PVMandatoryMappingFilter::p_type mf = (*it_lfmf)->clone<PVMandatoryMappingFilter>();
		mand_mapping_filters.push_back(mf);
	}
	std::vector<PVMandatoryMappingFilter::p_type>::const_iterator it_pmf;

	for (PVCol j = 0; j < get_column_count(); j++) {
		mandatory_param_map& params_map = _mapping->get_mandatory_params_for_col(j);
		tbb::tick_count tmap_start = tbb::tick_count::now();
		// Init the mandatory mapping
		for (it_pmf = mand_mapping_filters.begin(); it_pmf != mand_mapping_filters.end(); it_pmf++) {
			(*it_pmf)->set_dest_params(params_map);
			(*it_pmf)->set_decimal_type(_mapping_filters_rush[j]->get_decimal_type());
			(*it_pmf)->set_mapped(*this);
			(*it_pmf)->operator()(Picviz::mandatory_param_list_values(j, get_column_pointer(j)));
		}
		tbb::tick_count tmap_end = tbb::tick_count::now();

		PVLOG_INFO("(PVMapped::create_table) mandatory mapping for axis %d took %0.4f seconds.\n", j, (tmap_end-tmap_start).seconds());
	}

	// Validate all mapping!
	validate_all();

	// Clear mapping filters
	_mapping_filters_rush.clear();

	// Clear "group values" hash
	_grp_values_rush.clear();
}

void Picviz::PVMapped::process_rush_pipeline_chunk(PVCore::PVChunk const* chunk, PVRow const cur_r)
{
	PVCore::list_elts const& elts = chunk->c_elements();
	const PVRow new_size = cur_r + elts.size();
	if (new_size > _trans_table[0].size()) {
		// Reallocate everyone for 10 more millions
		for (mapped_row_t& mrow: _trans_table) {
			mrow.resize(mrow.size() + DEFAULT_MAPPING_NROWS);
		}
	}

	chunk->visit_by_column([&](PVRow const r, PVCol const c, PVCore::PVField const& field)
		{
			assert(c < (PVCol) _mapping_filters_rush.size());
			if (is_mapping_pure(c)) {
				// AG: HACK: if we are not the first mapping, that's a failure..
				this->_trans_table[c].at(r+cur_r) = field.mapped_value();
			}
			else {
				PVMappingFilter::p_type& mapping_filter = _mapping_filters_rush[c];
				this->_trans_table[c].at(r+cur_r) = mapping_filter->operator()(field);
			}
		});
}

/******************************************************************************
 *
 * Picviz::PVMapped::create_table
 *
 *****************************************************************************/
void Picviz::PVMapped::create_table()
{
	PVRush::PVNraw const& nraw = get_parent()->get_rushnraw();
	const PVRow nrows = nraw.get_number_rows();
	const PVCol ncols = nraw.get_number_cols();

	tbb::tick_count tstart = tbb::tick_count::now();
	allocate_table(nrows, ncols);

	PVLOG_INFO("(pvmapped::create_table) begin parallel mapping\n");

	// Create our own plugins from the library
	std::vector<PVMappingFilter::p_type> mapping_filters;
	mapping_filters.resize(ncols);
	for (PVCol j = 0; j < ncols; j++) {
		PVMappingFilter::p_type mf = _mapping->get_filter_for_col(j);
		if (mf) {
			mapping_filters[j] = mf->clone<PVMappingFilter>();
		}
	}

	// Do the same for the mandatory mappings
	std::vector<PVMandatoryMappingFilter::p_type> mand_mapping_filters;
	LIB_CLASS(Picviz::PVMandatoryMappingFilter)::list_classes const& lfmf = LIB_CLASS(Picviz::PVMandatoryMappingFilter)::get().get_list();
	LIB_CLASS(Picviz::PVMandatoryMappingFilter)::list_classes::const_iterator it_lfmf;
	mand_mapping_filters.reserve(lfmf.size());
	for (it_lfmf = lfmf.begin(); it_lfmf != lfmf.end(); it_lfmf++) {
		PVMandatoryMappingFilter::p_type mf = (*it_lfmf)->clone<PVMandatoryMappingFilter>();
		mand_mapping_filters.push_back(mf);
	}
	std::vector<PVMandatoryMappingFilter::p_type>::const_iterator it_pmf;

	try {
		// This is a hash whose key is "group_type", that contains the PVArgument
		// passed through all mapping filters that have the same group and type
		QHash<QString, PVCore::PVArgument> grp_values;
		for (PVCol j = 0; j < ncols; j++) {
			// Check that an update is required
			if (_mapping->get_properties_for_col(j).is_uptodate()) {
				continue;
			}

			// Get the corresponding object
			PVMappingFilter::p_type mapping_filter = mapping_filters[j];
			mandatory_param_map& params_map = _mapping->get_mandatory_params_for_col(j);
			params_map.clear();

			if (!mapping_filter) {
				PVLOG_ERROR("An invalid mapping type and/or mode is set for axis %d !\n", j);
				continue;
			}

			// Let's make our mapping
			mapping_filter->set_dest_array(nrows, get_column_pointer(j));
			//mapping_filter->set_axis(j, *get_format());
			// Get the group specific value if relevant
			QString group_key = _mapping->get_group_key_for_col(j);
			if (!group_key.isEmpty()) {
				PVCore::PVArgument& group_v = grp_values[group_key];
				mapping_filter->set_group_value(group_v);
			}
			boost::this_thread::interruption_point();
			tbb::tick_count tmap_start = tbb::tick_count::now();
			mapping_filter->operator()(j, nraw);
			tbb::tick_count tmap_end = tbb::tick_count::now();
			PVLOG_INFO("(PVMapped::create_table) parallel mapping for axis %d took %0.4f seconds.\n", j, (tmap_end-tmap_start).seconds());

			tmap_start = tbb::tick_count::now();
			// Init the mandatory mapping
			boost::this_thread::interruption_point();
			for (it_pmf = mand_mapping_filters.begin(); it_pmf != mand_mapping_filters.end(); it_pmf++) {
				(*it_pmf)->set_dest_params(params_map);
				(*it_pmf)->set_decimal_type(mapping_filter->get_decimal_type());
				(*it_pmf)->set_mapped(*this);
				(*it_pmf)->operator()(Picviz::mandatory_param_list_values(j, get_column_pointer(j)));
			}
			tmap_end = tbb::tick_count::now();

			PVLOG_INFO("(PVMapped::create_table) mandatory mapping for axis %d took %0.4f seconds.\n", j, (tmap_end-tmap_start).seconds());

			_mapping->set_uptodate_for_col(j);
			invalidate_plotted_children_column(j);
		}
		PVLOG_INFO("(pvmapped::create_table) end parallel mapping\n");
		tbb::tick_count tend = tbb::tick_count::now();
		PVLOG_INFO("(PVPlotted::create_table) mapping took %0.4f seconds.\n", (tend-tstart).seconds());
	}
	catch (boost::thread_interrupted const& e)
	{
		PVLOG_INFO("(PVPlotted::create_table) mapping canceled.\n");
		throw e;
	}
}

/******************************************************************************
 *
 * Picviz::PVMapped::to_csv
 *
 *****************************************************************************/
namespace Picviz { namespace __impl {
struct to_csv_value_holder
{
	template <typename T>
	static void call(Picviz::PVMapped::mapped_table_t const& trans_table, PVRow const i, PVCol const j)
	{
		std::cout << trans_table[j][i].storage_cast<T>();
	}
};
} }

void Picviz::PVMapped::to_csv()
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
 * Picviz::PVMapped::get_format
 *
 *****************************************************************************/
PVRush::PVFormat_p Picviz::PVMapped::get_format() const
{
	return get_parent()->get_rushnraw().get_format();
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_sub_col_minmax
 *
 *****************************************************************************/
namespace Picviz { namespace __impl {
struct get_sub_col_minmax_holder
{
	template <typename T>
	static void call(Picviz::PVMapped::mapped_sub_col_t& ret, Picviz::PVMapped::decimal_storage_type& min, Picviz::PVMapped::decimal_storage_type& max, Picviz::PVSelection const& sel, PVCol const col, Picviz::PVMapped::mapped_table_t const& trans_table)
	{
		min.set_max<T>();
		max.set_min<T>();

		const Picviz::PVMapped::decimal_storage_type* mapped_values = &trans_table[col][0];
		T& max_cast = max.storage_cast<T>();
		T& min_cast = min.storage_cast<T>();
		sel.visit_selected_lines([&](PVRow const i){
			const Picviz::PVMapped::decimal_storage_type v = mapped_values[i];
			const T v_cast = v.storage_cast<T>();
			if (v_cast > max_cast) {
				max_cast = v_cast;
			}
			if (v_cast < min_cast) {
				min_cast = v_cast;
			}
			ret.push_back(Picviz::PVMapped::mapped_sub_col_t::value_type(i, v));
		},
		trans_table[0].size());
	}
};
} }

void Picviz::PVMapped::get_sub_col_minmax(mapped_sub_col_t& ret, decimal_storage_type& min, decimal_storage_type& max, PVSelection const& sel, PVCol const col) const
{
	PVCore::DecimalType const type_col = get_decimal_type_of_col(col);
	PVRow size = get_row_count();
	ret.reserve(sel.get_number_of_selected_lines_in_range(0, size-1));

	decimal_storage_type::call_from_type<__impl::get_sub_col_minmax_holder>(type_col, ret, min, max, sel, col, _trans_table);
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_row_count
 *
 *****************************************************************************/
PVRow Picviz::PVMapped::get_row_count() const
{
	return get_parent()->get_row_count();
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_column_count
 *
 *****************************************************************************/
PVCol Picviz::PVMapped::get_column_count() const
{
	return _trans_table.size();
}

void Picviz::PVMapped::add_column(PVMappingProperties const& props)
{
	_mapping->add_column(props);
}

void Picviz::PVMapped::process_parent_source()
{
	create_table();
}

void Picviz::PVMapped::process_from_parent_source()
{
	process_parent_source();
	// Process plotting children
	for (auto plotted_p : get_children<PVPlotted>()) {
		plotted_p->process_from_parent_mapped();
	}
}

void Picviz::PVMapped::invalidate_plotted_children_column(PVCol j)
{
	for (auto plotted_p : get_children<PVPlotted>()) {
		plotted_p->invalidate_column(j);
	}
}

void Picviz::PVMapped::invalidate_all()
{
	_mapping->invalidate_all();
}

void Picviz::PVMapped::validate_all()
{
	_mapping->validate_all();
}

QList<PVCol> Picviz::PVMapped::get_columns_indexes_values_within_range(decimal_storage_type const min, decimal_storage_type const max, double rate)
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_column_count();
	QList<PVCol> cols_ret;

#if 0
	if (min > max) {
		return cols_ret;
	}

	double nrows_d = (double) nrows;
	for (PVCol j = 0; j < ncols; j++) {
		PVRow nmatch = 0;
		const float* values = trans_table.getRowData(j);
		// TODO: optimise w/ SIMD if relevant
		for (PVRow i = 0; i < nrows; i++) {
			const float v = values[i];
			if (v >= min && v <= max) {
				nmatch++;
			}
		}
		if ((double)nmatch/nrows_d >= rate) {
			cols_ret << j;
		}
	}
#endif

	return cols_ret;
}

QList<PVCol> Picviz::PVMapped::get_columns_indexes_values_not_within_range(decimal_storage_type const min, decimal_storage_type const max, double rate)
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_column_count();
	QList<PVCol> cols_ret;

#if 0

	if (min > max) {
		return cols_ret;
	}

	double nrows_d = (double) nrows;
	for (PVCol j = 0; j < ncols; j++) {
		PVRow nmatch = 0;
		const float* values = trans_table.getRowData(j);
		// TODO: optimise w/ SIMD if relevant
		for (PVRow i = 0; i < nrows; i++) {
			const float v = values[i];
			if (v < min || v > max) {
				nmatch++;
			}
		}
		if ((double)nmatch/nrows_d >= rate) {
			cols_ret << j;
		}
	}
#endif

	return cols_ret;
}

void Picviz::PVMapped::serialize_write(PVCore::PVSerializeObject& so)
{
	data_tree_mapped_t::serialize_write(so);

	so.object(QString("mapping"), *_mapping, QString(), false, (PVMapping*) NULL, false);
}

void Picviz::PVMapped::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v)
{
	PVMapping* mapping = new PVMapping();
	so.object(QString("mapping"), *mapping, QString(), false, (PVMapping*) NULL, false);
	_mapping = PVMapping_p(mapping);
	_mapping->set_mapped(this);

	// It important to deserialize the children after the mapping otherwise PVPlottingProperties complains that there is no mapping!
	data_tree_mapped_t::serialize_read(so, v);
}
