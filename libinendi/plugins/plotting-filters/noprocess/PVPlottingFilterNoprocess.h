/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVPLOTTINGFILTERNOPROCESS_H
#define PVFILTER_PVPLOTTINGFILTERNOPROCESS_H

#include <inendi/PVPlottingFilter.h>

namespace Inendi
{

class PVPlottingFilterNoprocess : public PVPlottingFilter
{
  public:
	uint32_t* operator()(PVCore::PVDecimalStorage<32> const* values) override;
	QString get_human_name() const override { return QString("Default"); }

	CLASS_FILTER(PVPlottingFilterNoprocess)
};
}

#endif
