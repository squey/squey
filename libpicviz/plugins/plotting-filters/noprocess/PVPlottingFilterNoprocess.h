/**
 * \file PVPlottingFilterNoprocess.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVFILTER_PVPLOTTINGFILTERNOPROCESS_H
#define PVFILTER_PVPLOTTINGFILTERNOPROCESS_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterNoprocess: public PVPlottingFilter
{
public:
	uint32_t* operator()(mapped_decimal_storage_type const* values) override;
	QString get_human_name() const override { return QString("Default"); }

	CLASS_FILTER(PVPlottingFilterNoprocess)
};

}

#endif
