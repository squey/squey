/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERTIMEWEEK_H
#define PVFILTER_PVMAPPINGFILTERTIMEWEEK_H

#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>
#include "../default/PVMappingFilterTimeDefault.h"

namespace Inendi {

class PVMappingFilterTimeWeek: public PVMappingFilterTimeDefault
{
protected:
	int32_t cal_to_int(Calendar* cal, bool& success) override;
	QString get_human_name() const override { return QString("Week"); }

	CLASS_REGISTRABLE(PVMappingFilterTimeWeek)
};

}

#endif
