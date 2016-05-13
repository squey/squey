/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVPLOTTINGFILTERINTEGERPORT_H
#define PVFILTER_PVPLOTTINGFILTERINTEGERPORT_H

#include <inendi/PVPlottingFilter.h>

namespace Inendi
{

class PVPlottingFilterIntegerPort : public PVPlottingFilter
{
  public:
	uint32_t* operator()(mapped_decimal_storage_type const* values) override;
	QString get_human_name() const { return QString("TCP/UDP port"); }

	CLASS_FILTER(PVPlottingFilterIntegerPort)
};
}

#endif
