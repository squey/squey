//! \file PVMappingFilter.h
//! $Id: PVMandatoryMappingFilter.h 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMANDATORYMAPPINGFILTER_H
#define PVFILTER_PVMANDATORYMAPPINGFILTER_H

#include <pvcore/general.h>
#include <pvfilter/PVFilterFunction.h>
#include <pvfilter/PVFilterLibrary.h>
#include <pvrush/PVNraw.h>

#include <tbb/concurrent_unordered_map.h>
#include <utility>

namespace Picviz {

typedef std::pair<const PVRush::PVNraw::nraw_table_line*, float*> mandatory_param_list_values;
typedef std::pair<QString, float> mandatory_param_value;

// This defines the different parameters that a mandatory mapping
// filter can set for an axis
enum mandatory_mapping_param_type
{
	mandatory_ymin,
	mandatory_ymax
};

typedef tbb::concurrent_unordered_map<mandatory_mapping_param_type, mandatory_param_value> mandatory_param_map;

class LibExport PVMandatoryMappingFilter: public PVFilter::PVFilterFunctionRegistrable<int, mandatory_param_list_values const&, PVMandatoryMappingFilter>
{
public:
	typedef PVFilter::PVFilterFunctionRegistrable<int, mandatory_param_list_values const&, PVMandatoryMappingFilter>::base_registrable base_registrable;

public:
	typedef PVMandatoryMappingFilter FilterT;
	typedef boost::shared_ptr<PVMandatoryMappingFilter> p_type;

public:
	PVMandatoryMappingFilter();

public:
	virtual int operator()(mandatory_param_list_values const& values);

	virtual int operator()(mandatory_param_value const& value);
	virtual void init_from_first(mandatory_param_value const& value);

	void set_dest_params(mandatory_param_map& params);
protected:
	mandatory_param_map* _mandatory_params;
};

typedef PVMandatoryMappingFilter::func_type PVMandatoryMappingFilter_f;

}

#ifdef WIN32
picviz_FilterLibraryDecl PVFilter::PVFilterLibrary<Picviz::PVMandatoryMappingFilter::FilterT>;
#endif

#endif
