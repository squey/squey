/**
 * \file PVPlottingFilterTimeDefault.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVFILTER_PVPLOTTINGFILTERTIMEDEFAULT_H
#define PVFILTER_PVPLOTTINGFILTERTIMEDEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterTimeDefault: public PVPlottingFilter
{
public:
	uint32_t* operator()(mapped_decimal_storage_type const* value) override;
	QString get_human_name() const override { return QString("Default (depends on mapping)"); }

	CLASS_FILTER(PVPlottingFilterTimeDefault)
};

}

#endif
