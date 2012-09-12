/**
 * \file PVPlottingFilterIntegerPort.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVFILTER_PVPLOTTINGFILTERINTEGERPORT_H
#define PVFILTER_PVPLOTTINGFILTERINTEGERPORT_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterIntegerPort: public PVPlottingFilter
{
public:
	uint32_t* operator()(mapped_decimal_storage_type const* values) override;
	QString get_human_name() const { return QString("TCP/UDP port"); }

	CLASS_FILTER(PVPlottingFilterIntegerPort)
};

}

#endif
