/**
 * \file PVFieldsMappingFilter.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

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
	PVCore::list_fields& many_to_many(PVCore::list_fields& fields);
protected:
	map_filters _mfilters;
};

}

#endif
