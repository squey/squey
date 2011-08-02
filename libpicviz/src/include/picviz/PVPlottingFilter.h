//! \file PVMappingFilter.h
//! $Id: PVPlottingFilter.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVPLOTTINGFILTER_H
#define PVFILTER_PVPLOTTINGFILTER_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/filter/PVFilterLibrary.h>
#include <picviz/PVMandatoryMappingFilter.h>

namespace Picviz {

class LibPicvizDecl PVPlottingFilter: public PVFilter::PVFilterFunctionRegistrable<float*, float*, PVPlottingFilter>
{
public:
	typedef PVFilter::PVFilterFunctionRegistrable<float*, float*, PVPlottingFilter>::base_registrable base_registrable;

public:
	typedef PVPlottingFilter FilterT;
	typedef boost::shared_ptr<PVPlottingFilter> p_type;
public:
	PVPlottingFilter();

public:
	virtual float* operator()(float* value);
	virtual float operator()(float value);

	void set_dest_array(PVRow size, float* arr);
	void set_mapping_mode(QString const& mapping_mode);
	void set_mandatory_params(Picviz::mandatory_param_map const& params);
protected:
	QString _mapping_mode;
	PVRow _dest_size;
	float* _dest;
	Picviz::mandatory_param_map const* _mandatory_params;
};

typedef PVPlottingFilter::func_type PVPlottingFilter_f;

}

#ifdef WIN32
LibPicvizDeclExplicitTempl PVFilter::PVFilterLibrary<Picviz::PVPlottingFilter::FilterT>;
#endif

#endif
