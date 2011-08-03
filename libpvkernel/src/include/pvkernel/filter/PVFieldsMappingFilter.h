//! \file PVFieldsMappingFilter.h
//! $Id: PVFieldsMappingFilter.h 3165 2011-06-16 05:05:39Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVFIELDSMAPPINGFILTER_H
#define PVFILTER_PVFIELDSMAPPINGFILTER_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilter.h>

#include <vector>
#include <map>

namespace PVFilter {

class LibKernelDecl PVFieldsMappingFilter : public PVFieldsFilter<many_to_many> {
public:
	typedef std::vector<chunk_index> list_indexes;
	typedef std::map<list_indexes, PVFieldsBaseFilter_f> map_filters;
public:
	PVFieldsMappingFilter(map_filters const& mfilters);
public:
	PVCore::list_fields& operator()(PVCore::list_fields& fields);
protected:
	map_filters _mfilters;

	CLASS_FILTER(PVFilter::PVFieldsMappingFilter)
};

}

#endif
