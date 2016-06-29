/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVPLOTTINGFILTERMINMAX_H
#define PVFILTER_PVPLOTTINGFILTERMINMAX_H

#include <inendi/PVPlottingFilter.h>

namespace Inendi
{

class PVPlottingFilterMinmax : public PVPlottingFilter
{
  public:
	uint32_t* operator()(pvcop::db::array const& mapped) override;
	QString get_human_name() const override { return QString("Min/max"); }

	std::set<std::pair<std::string, std::string>> list_usable_type() const override { return {}; }

  private:
	CLASS_FILTER(PVPlottingFilterMinmax)
};
}

#endif
