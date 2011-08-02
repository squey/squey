//! \file PVMappingFilterIPv4Default.h
//! $Id: PVMappingFilterIPv4Default.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMandatoryMappingFilter.h>
#include <tbb/concurrent_unordered_map.h>

#include <QString>

// Used by the concurrennt hash map below
size_t tbb_hasher(const QString& str);

namespace Picviz {

class PVMandatoryMappingFilterMinmax : public PVMandatoryMappingFilter
{
public:
	PVMandatoryMappingFilterMinmax();

public:
	int operator()(mandatory_param_list_values const& values);

protected:

	CLASS_FILTER(PVMandatoryMappingFilterMinmax)
};

}

#endif
