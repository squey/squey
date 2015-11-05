/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERDATEMONTH_H
#define PVFILTER_PVMAPPINGFILTERDATEMONTH_H

#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>
#include "../PVMappingFilterDateBase.h"

namespace Inendi {

class PVMappingFilterDateMonth : public PVMappingFilterDateBase
{
public:
	PVMappingFilterDateMonth();

protected:
	QString get_human_name() const override { return QString("Month"); }

	CLASS_REGISTRABLE(PVMappingFilterDateMonth)
};

}

#endif // PVFILTER_PVMAPPINGFILTERDATEMONTH_H
