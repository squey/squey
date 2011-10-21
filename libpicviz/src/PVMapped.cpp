//! \file PVMapped.cpp
//! $Id: PVMapped.cpp 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QList>
#include <QStringList>
#include <QString>

#include <pvkernel/rush/PVFormat.h>

#include <picviz/PVMandatoryMappingFunction.h>
#include <picviz/PVMandatoryMappingFilter.h>
#include <picviz/PVMapping.h>
#include <picviz/PVMapped.h>
#include <picviz/PVSource.h>

#include <picviz/PVRoot.h>

#include <iostream>

#include <omp.h>

/******************************************************************************
 *
 * Picviz::PVMapped::PVMapped
 *
 *****************************************************************************/
Picviz::PVMapped::PVMapped(PVMapping_p parent)
{
	mapping = parent;
	root = parent->root;

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

/******************************************************************************
 *
 * Picviz::PVMapped::create_table
 *
 *****************************************************************************/
int Picviz::PVMapped::create_table()
{
	PVRush::PVNraw::nraw_table& qt_nraw = get_qtnraw();

	const PVRow nrows = (PVRow)qt_nraw.size();
	const PVCol ncols = (PVCol) qt_nraw.at(0).size();
	table.reserve(ncols,nrows);
	
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
	
	PVRush::PVNraw::nraw_trans_table const& trans_nraw = get_trans_nraw();
	PVLOG_INFO("(pvmapped::create_table) begin parallel mapping\n");

	// Create our own plugins from the library
	std::vector<PVMappingFilter::p_type> mapping_filters;
	mapping_filters.resize(ncols);
	for (PVCol j = 0; j < ncols; j++) {
		PVMappingFilter::p_type mf = mapping->get_filter_for_col(j);
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

	// This is a hash whose key is "group_type", that contains the PVArgument
	// passed through all mapping filters that have the same group and type
	QHash<QString, PVCore::PVArgument> grp_values;
	for (PVCol j = 0; j < ncols; j++) {
		// Get the corresponding object
		PVRush::PVNraw::nraw_table_line const& fields = trans_nraw[j];
		PVMappingFilter::p_type mapping_filter = mapping_filters[j];
		mandatory_param_map& params_map = mapping->get_mandatory_params_for_col(j);

		if (!mapping_filter) {
			PVLOG_ERROR("An invalid mapping type and/or mode is set for axis %d !\n", j);
			continue;
		}

		// Let's make our mapping
		mapping_filter->set_dest_array(nrows, trans_table.getRowData(j));
		mapping_filter->set_format(j, *get_format());
		// Get the group specific value if relevant
		QString group_key = mapping->get_group_key_for_col(j);
		if (!group_key.isEmpty()) {
			PVCore::PVArgument& group_v = grp_values[group_key];
			mapping_filter->set_group_value(group_v);
		}
		tbb::tick_count tmap_start = tbb::tick_count::now();
		mapping_filter->operator()(fields);
		tbb::tick_count tmap_end = tbb::tick_count::now();
		PVLOG_INFO("(PVMapped::create_table) parallel mapping for axis %d took %0.4f seconds.\n", j, (tmap_end-tmap_start).seconds());
		
		tmap_start = tbb::tick_count::now();
		// Init the mandatory mapping
		for (it_pmf = mand_mapping_filters.begin(); it_pmf != mand_mapping_filters.end(); it_pmf++) {
			(*it_pmf)->set_dest_params(params_map);
			(*it_pmf)->operator()(Picviz::mandatory_param_list_values(&fields, trans_table.getRowData(j)));
		}
		tmap_end = tbb::tick_count::now();

		PVLOG_INFO("(PVMapped::create_table) mandatory mapping for axis %d took %0.4f seconds.\n", j, (tmap_end-tmap_start).seconds());
	}
	PVLOG_INFO("(pvmapped::create_table) end parallel mapping\n");
#endif
	

	tbb::tick_count tend = tbb::tick_count::now();
	PVLOG_INFO("(PVPlotted::create_table) mapping took %0.4f seconds.\n", (tend-tstart).seconds());

	// Free the transposed NRAW
	clear_trans_nraw();
	
	return 0;
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
			if (j!=table.getWidth()-1) {
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
	return mapping->source->get_rushnraw().format;
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_qtnraw
 *
 *****************************************************************************/
PVRush::PVNraw::nraw_table& Picviz::PVMapped::get_qtnraw()
{
	return mapping->get_qtnraw();
}

const PVRush::PVNraw::nraw_table& Picviz::PVMapped::get_qtnraw() const
{
	return mapping->get_qtnraw();
}

const PVRush::PVNraw::nraw_trans_table& Picviz::PVMapped::get_trans_nraw() const
{
	return mapping->get_trans_nraw();
}

void Picviz::PVMapped::clear_trans_nraw()
{
	mapping->clear_trans_nraw();
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_source_parent
 *
 *****************************************************************************/
Picviz::PVSource* Picviz::PVMapped::get_source_parent()
{
	return mapping->get_source_parent();
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_root
 *
 *****************************************************************************/
Picviz::PVRoot* Picviz::PVMapped::get_root()
{
	return mapping->root;
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_row_count
 *
 *****************************************************************************/
PVRow Picviz::PVMapped::get_row_count()
{
	return table.getHeight();
}

/******************************************************************************
 *
 * Picviz::PVMapped::get_column_count
 *
 *****************************************************************************/
PVCol Picviz::PVMapped::get_column_count()
{
	return table.getWidth();
}
