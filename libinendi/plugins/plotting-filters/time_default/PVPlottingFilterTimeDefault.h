/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVPLOTTINGFILTERTIMEDEFAULT_H
#define PVFILTER_PVPLOTTINGFILTERTIMEDEFAULT_H

#include <inendi/PVPlottingFilter.h>

namespace Inendi
{

class PVPlottingFilterTimeDefault : public PVPlottingFilter
{
  public:
	uint32_t* operator()(pvcop::db::array const& mapped) override;

	QString get_human_name() const override { return QString("Default (depends on mapping)"); }

	CLASS_FILTER(PVPlottingFilterTimeDefault)
};
}

#endif
