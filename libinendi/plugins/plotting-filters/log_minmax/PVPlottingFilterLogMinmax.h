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
	void operator()(pvcop::db::array const& mapped,
	                pvcop::db::array const& minmax,
	                pvcop::core::array<value_type>& dest) override;
	QString get_human_name() const override { return QString("Logarithmic min/max"); }

	std::set<plotting_capability> list_usable_type() const override { return {}; }

  private:
	CLASS_FILTER_NOPARAM(PVPlottingFilterLogMinmax)
};
}

#endif
