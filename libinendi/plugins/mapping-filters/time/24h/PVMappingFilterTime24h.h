/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERTIME24H_H
#define PVFILTER_PVMAPPINGFILTERTIME24H_H

#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>
#include "../default/PVMappingFilterTimeDefault.h"

namespace Inendi {

class PVMappingFilterTime24h: public PVMappingFilterTimeDefault
{
public:
	QString get_human_name() const override { return QString("24 hours"); }
protected:
	int32_t cal_to_int(Calendar* cal, bool& success) override;

	CLASS_REGISTRABLE(PVMappingFilterTime24h)
};

}

#endif
