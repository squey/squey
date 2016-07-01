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

class PVPlottingFilterEnum : public PVPlottingFilter
{
  public:
	uint32_t* operator()(pvcop::db::array const& mapped, pvcop::db::array const& minmax) override;
	QString get_human_name() const override { return QString("Uniform"); }

	std::set<std::pair<std::string, std::string>> list_usable_type() const override { return {}; }

	CLASS_FILTER(PVPlottingFilterEnum)
};
}

#endif
