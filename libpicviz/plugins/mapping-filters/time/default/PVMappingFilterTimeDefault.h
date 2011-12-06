//! \file PVMappingFilterIPv4Default.h
//! $Id: PVMappingFilterIPv4Default.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMAPPINGFILTERTIMEDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERTIMEDEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>

#include <unicode/calendar.h>

namespace Picviz {

class PVMappingFilterTimeDefault: public PVMappingFilter
{
public:
	PVMappingFilterTimeDefault(PVCore::PVArgumentList const& args = PVMappingFilterTimeDefault::default_args());

public:
	float* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values);
	QString get_human_name() const { return QString("Default"); }
protected:
	virtual float cal_to_float(Calendar* cal, bool& success);

	CLASS_FILTER(PVMappingFilterTimeDefault)
};

}

#endif
