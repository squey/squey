/**
 * \file PVMandatoryMappingFilter.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMANDATORYMAPPINGFILTER_H
#define PVFILTER_PVMANDATORYMAPPINGFILTER_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/rush/PVNraw.h>

#include <picviz/PVMapped_types.h>

//#include <tbb/concurrent_unordered_map.h>
#include <map>
#include <utility>

namespace Picviz {

typedef std::pair<PVCol, mapped_decimal_storage_type*> mandatory_param_list_values;
typedef std::pair<QString, mapped_decimal_storage_type> mandatory_param_value;

class PVMapped;

// This defines the different parameters that a mandatory mapping
// filter can set for an axis
enum mandatory_mapping_param_type
{
	mandatory_ymin,
	mandatory_ymax
};

//typedef tbb::concurrent_unordered_map<mandatory_mapping_param_type, mandatory_param_value> mandatory_param_map;
typedef std::map<mandatory_mapping_param_type, mandatory_param_value> mandatory_param_map;

class LibPicvizDecl PVMandatoryMappingFilter: public PVFilter::PVFilterFunctionBase<int, mandatory_param_value const&>, public PVCore::PVRegistrableClass<PVMandatoryMappingFilter>
{
public:
	typedef boost::shared_ptr<PVMandatoryMappingFilter> p_type;
	typedef PVMandatoryMappingFilter FilterT;
	typedef Picviz::mapped_decimal_storage_type decimal_storage_type;

public:
	PVMandatoryMappingFilter();

public:
	virtual int operator()(mandatory_param_list_values const& values);

	void set_dest_params(mandatory_param_map& params);
	void set_decimal_type(PVCore::DecimalType decimal_type) { _decimal_type = decimal_type; }
	void set_mapped(Picviz::PVMapped const& mapped) { _mapped = &mapped; }
protected:
	mandatory_param_map* _mandatory_params;
	PVCore::DecimalType _decimal_type;
	PVMapped const* _mapped;
};

typedef PVMandatoryMappingFilter::func_type PVMandatoryMappingFilter_f;

}

#ifdef WIN32
LibPicvizDeclExplicitTempl PVCore::PVClassLibrary<Picviz::PVMandatoryMappingFilter>;
LibPicvizDeclExplicitTempl PVCore::PVTag<Picviz::PVMandatoryMappingFilter>;
#endif

#endif
