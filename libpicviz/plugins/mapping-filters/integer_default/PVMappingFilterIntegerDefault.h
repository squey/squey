/**
 * \file PVMappingFilterIntegerDefault.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERINTEGER_H
#define PVFILTER_PVMAPPINGFILTERINTEGER_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>

namespace Picviz {

class PVMappingFilterIntegerDefault: public PVMappingFilter
{
public:
	float operator()(QString const& str);
	QString get_human_name() const { return QString("default"); }

	CLASS_FILTER(PVMappingFilterIntegerDefault)
};

}

#endif
