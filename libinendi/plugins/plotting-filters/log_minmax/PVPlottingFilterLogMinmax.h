/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVPLOTTINGFILTERLOGMINMAX_H
#define PVFILTER_PVPLOTTINGFILTERLOGMINMAX_H

#include <inendi/PVPlottingFilter.h>

namespace Inendi
{

class PVPlottingFilterLogMinmax : public PVPlottingFilter
{
  public:
	PVPlottingFilterLogMinmax(
	    PVCore::PVArgumentList const& args = PVPlottingFilterLogMinmax::default_args());

  public:
	uint32_t* operator()(pvcop::db::array const& mapped) override;
	QString get_human_name() const override { return QString("Logarithmic min/max"); }

  private:
	CLASS_FILTER(PVPlottingFilterLogMinmax)
};
}

#endif
