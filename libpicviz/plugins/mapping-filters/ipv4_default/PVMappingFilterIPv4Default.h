//! \file PVMappingFilterIPv4Default.h
//! $Id: PVMappingFilterIPv4Default.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

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
