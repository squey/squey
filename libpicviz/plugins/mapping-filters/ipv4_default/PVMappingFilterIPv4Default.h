/**
 * \file PVMappingFilterIPv4Default.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERIPV4DEFAULT_H
#define PVFILTER_PVMAPPINGFILTERIPV4DEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>

namespace Picviz {

class PVMappingFilterIPv4Default: public PVMappingFilter
{
public:
	float operator()(QString const& str);
	QString get_human_name() const { return QString("Default"); }

	CLASS_FILTER(PVMappingFilterIPv4Default)
};

}

#endif
