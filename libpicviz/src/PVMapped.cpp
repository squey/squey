//! \file PVMapped.cpp
//! $Id: PVMapped.cpp 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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

#include <picviz/PVRoot.h>

#include <boost/thread.hpp>

#include <iostream>

#include <float.h>
#include <omp.h>

/******************************************************************************
 *
 * Picviz::PVMapped::PVMapped
 *
 *****************************************************************************/
Picviz::PVMapped::PVMapped(PVMapping const& mapping)
{
	set_mapping(mapping);
	create_table();
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

void Picviz::PVMapped::set_mapping(PVMapping const& mapping)
{
	_mapping = mapping;
	_root = _mapping.get_root_parent();
	_source = _mapping.get_source_parent();
}

/******************************************************************************
 *
 * Picviz::PVMapped::create_table
 *
 *****************************************************************************/
void Picviz::PVMapped::create_table()
{
	PVRush::PVNraw::nraw_table& qt_nraw = get_qtnraw();

	const PVRow nrows = (PVRow)qt_nraw.get_nrows();
	const PVCol ncols = (PVCol) qt_nraw.get_ncols();
	
	tbb::tick_count tstart = tbb::tick_count::now();
	trans_table.reserve(nrows,ncols);

#ifdef CUDA
PVLOG_INFO("(pvmapped::create_table) begin cuda mapping\n");
	{
    
	    //for all columns except the second
	    for (PVCol j = 0; j < ncols; j++) {//for each col...
		for (PVRow i = 0; i < nrows; ++i) {//for each rows
		    
		    if(j!=1&&j!=2&&j!=4&&j!=5){
		    //if(j!=1){
		      QStringList slist = qt_nraw.at(i);
		      QString value = slist[j];
		      float val = mapping->get_position(j, value);
		      table.setValue(val,i,j);
		      ////trans_table.setValue(val, j, i);//seg fault
		      ////run_mandatory_mapping(this->get_root(), i, j, value, val, is_first, userdata);
		    }else{//if(j!=1&&j!=2&&j!=4&&j!=5)
			//parallele making
		    }//else : if(j!=1&&j!=2&&j!=4&&j!=5)
		    
		    is_first = 0;
		}
	    }
	    
	    //cuda call to process the second column
	    ///FIXME for testing
	    for(PVCol j = 0; j < ncols; j++){
		if(j!=1&&j!=2&&j!=4&&j!=5){
		//if(j!=1){
		    //linear making
		}else{//if(j!=1&&j!=2&&j!=4&&j!=5)
		    //data transformed
		    char *cuda_host_nraw = (char*)malloc(nrows*1000*sizeof(char));///FIXME memorie leak
		    
		    PVLOG_INFO("START DATA TRANSFORM\n");
		    
		    for(uint i=0;i<nrows;i++){
		      QStringList slist = qt_nraw.at(i);
		      QString value = slist[j];		
		      //data transform
		      strcpy(&cuda_host_nraw[1000*i],value.toStdString().c_str());

		    }
		    //kenel call
		    pvmapped_create_table_enum_default(j, cuda_host_nraw, nrows, &table);

		}//else : if(j!=1&&j!=2&&j!=4&&j!=5)
	      
	    }
	    
	}
	PVLOG_INFO("(pvmapped::create_table) end cuda mapping\n");
#else

	// This will use the trans_table of the nraw
	
	/*
	_source->get_rushnraw().create_trans_nraw();	
	PVRush::PVNraw::nraw_trans_table const& trans_nraw = get_trans_nraw();
	*/
	PVRush::PVNraw const& nraw = _source->get_rushnraw();

	PVLOG_INFO("(pvmapped::create_table) begin parallel mapping\n");

	// Create our own plugins from the library
	std::vector<PVMappingFilter::p_type> mapping_filters;
	mapping_filters.resize(ncols);
	for (PVCol j = 0; j < ncols; j++) {
		PVMappingFilter::p_type mf = _mapping.get_filter_for_col(j);
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
			if (_mapping.get_properties_for_col(j).is_uptodate()) {
				continue;
			}

			// Get the corresponding object
			PVRush::PVNraw::const_trans_nraw_table_line fields = nraw.get_col(j);
			PVMappingFilter::p_type mapping_filter = mapping_filters[j];
			mandatory_param_map& params_map = _mapping.get_mandatory_params_for_col(j);
			params_map.clear();

			if (!mapping_filter) {
				PVLOG_ERROR("An invalid mapping type and/or mode is set for axis %d !\n", j);
				continue;
			}

			// Let's make our mapping
			mapping_filter->set_dest_array(nrows, trans_table.getRowData(j));
			//mapping_filter->set_axis(j, *get_format());
			// Get the group specific value if relevant
			QString group_key = _mapping.get_group_key_for_col(j);
			if (!group_key.isEmpty()) {
				PVCore::PVArgument& group_v = grp_values[group_key];
				mapping_filter->set_group_value(group_v);
			}
			boost::this_thread::interruption_point();
			tbb::tick_count tmap_start = tbb::tick_count::now();
			mapping_filter->operator()(fields);
			tbb::tick_count tmap_end = tbb::tick_count::now();
			PVLOG_INFO("(PVMapped::create_table) parallel mapping for axis %d took %0.4f seconds.\n", j, (tmap_end-tmap_start).seconds());

			tmap_start = tbb::tick_count::now();
			// Init the mandatory mapping
			boost::this_thread::interruption_point();
			for (it_pmf = mand_mapping_filters.begin(); it_pmf != mand_mapping_filters.end(); it_pmf++) {
				(*it_pmf)->set_dest_params(params_map);
				(*it_pmf)->operator()(Picviz::mandatory_param_list_values(&fields, trans_table.getRowData(j)));
			}
			tmap_end = tbb::tick_count::now();

			PVLOG_INFO("(PVMapped::create_table) mandatory mapping for axis %d took %0.4f seconds.\n", j, (tmap_end-tmap_start).seconds());

			_mapping.set_uptodate_for_col(j);
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
#endif
	
	// Free the transposed NRAW
	//clear_trans_nraw();
}

/******************************************************************************
 *
 * Picviz::PVMapped::to_csv
 *
 *****************************************************************************/
void Picviz::PVMapped::to_csv()
{
	// WARNING: this is all but efficient. Uses this for testing and
	// debugging purpose only !
	for (PVRow i = 0; i < (PVRow) trans_table.getWidth(); i++) {
		for (PVCol j = 0; j < (PVCol) trans_table.getHeight(); j++) {
			std::cout << trans_table.getValue(j,i);
			if (j!=trans_table.getHeight()-1) {
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
PVRush::PVFormat_p Picviz::PVMapped::get_format()
{
	return _mapping.get_source_parent()->get_rushnraw().format;
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_sub_col_minmax
 *
 *****************************************************************************/
void Picviz::PVMapped::get_sub_col_minmax(mapped_sub_col_t& ret, float& min, float& max, PVSelection const& sel, PVCol col) const
{
	min = FLT_MAX;
	max = 0;
	PVRow size = get_qtnraw().get_nrows();
	ret.reserve(sel.get_number_of_selected_lines_in_range(0, size-1));
	const float* mapped_values = trans_table.getRowData(col);
	for (PVRow i = 0; i < size; i++) {
		if (sel.get_line(i)) {
			const float v = mapped_values[i];
			if (v > max) {
			   max = v;
			}
	 		if (v < min) {
				min = v;
			}		
			ret.push_back(mapped_sub_col_t::value_type(i, v));
		}
	}
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_qtnraw
 *
 *****************************************************************************/
PVRush::PVNraw::nraw_table& Picviz::PVMapped::get_qtnraw()
{
	return _mapping.get_qtnraw();
}

const PVRush::PVNraw::nraw_table& Picviz::PVMapped::get_qtnraw() const
{
	return _mapping.get_qtnraw();
}

const PVRush::PVNraw::nraw_trans_table& Picviz::PVMapped::get_trans_nraw() const
{
	return _mapping.get_trans_nraw();
}

void Picviz::PVMapped::clear_trans_nraw()
{
	_mapping.clear_trans_nraw();
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_source_parent
 *
 *****************************************************************************/
Picviz::PVSource* Picviz::PVMapped::get_source_parent()
{
	return _mapping.get_source_parent();
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_root_parent
 *
 *****************************************************************************/
Picviz::PVRoot* Picviz::PVMapped::get_root_parent()
{
	return _mapping.get_root_parent();
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_row_count
 *
 *****************************************************************************/
PVRow Picviz::PVMapped::get_row_count()
{
	return trans_table.getWidth();
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_column_count
 *
 *****************************************************************************/
PVCol Picviz::PVMapped::get_column_count()
{
	return trans_table.getHeight();
}

void Picviz::PVMapped::add_column(PVMappingProperties const& props)
{
	_mapping.add_column(props);
}

void Picviz::PVMapped::add_plotted(PVPlotted_p plotted)
{
	_plotteds.push_back(plotted);
}

void Picviz::PVMapped::process_from_source(PVSource* src, bool keep_views_info)
{
	_mapping.set_source(src);
	set_mapping(_mapping);

	process_from_parent_source(keep_views_info);
}

void Picviz::PVMapped::process_parent_source()
{
	create_table();
}

void Picviz::PVMapped::process_from_parent_source(bool keep_views_info)
{
	process_parent_source();
	// Process plotting children
	for (int i = 0; i < _plotteds.size(); i++) {
		PVPlotted_p plotted = _plotteds[i];
		plotted->process_from_mapped(this, keep_views_info);
	}
}

void Picviz::PVMapped::invalidate_plotted_children_column(PVCol j)
{
	list_plotted_t::iterator it;
	for (it = _plotteds.begin(); it != _plotteds.end(); it++) {
		PVPlotted_p plotted = *it;
		plotted->invalidate_column(j);
	}
}

void Picviz::PVMapped::invalidate_all()
{
	_mapping.invalidate_all();
}

QList<PVCol> Picviz::PVMapped::get_columns_indexes_values_within_range(float min, float max, double rate)
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_column_count();
	QList<PVCol> cols_ret;

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

	return cols_ret;
}

QList<PVCol> Picviz::PVMapped::get_columns_indexes_values_not_within_range(float min, float max, double rate)
{
	const PVRow nrows = get_row_count();
	const PVCol ncols = get_column_count();
	QList<PVCol> cols_ret;

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

	return cols_ret;
}

void Picviz::PVMapped::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.object(QString("mapping"), _mapping, QString(), false, (PVMapping*) NULL, false);
	QStringList plotted_names;
	list_plotted_t::const_iterator it;
	for (it = _plotteds.begin(); it != _plotteds.end(); it++) {
		plotted_names << (*it)->get_name();
	}
	so.list("plotted", _plotteds, "Plottings", (PVPlotted*) NULL, plotted_names, true, true);

	if (so.is_writing()) {
		unsigned int size = trans_table.count();
		so.attribute("data-size", size);
		so.buffer("data", trans_table.getData(), size*sizeof(float));
	}
	else {
		unsigned int size = 0;
		so.attribute("data-size", size);
		trans_table.reserve(_mapping.get_number_cols(), size/_mapping.get_number_cols());
		so.buffer("data", trans_table.getData(), size*sizeof(float));
	}
}
