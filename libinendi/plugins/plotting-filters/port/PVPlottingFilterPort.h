/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVPLOTTINGFILTERPORT_H
#define PVFILTER_PVPLOTTINGFILTERPORT_H

#include <inendi/PVPlottingFilter.h>

namespace Inendi
{

class PVPlottingFilterPort : public PVPlottingFilter
{
  public:
	void operator()(pvcop::db::array const& mapped,
	                pvcop::db::array const& minmax,
	                uint32_t* dest) override;
	QString get_human_name() const { return QString("TCP/UDP port"); }

	std::set<plotting_capability> list_usable_type() const override
	{
		return {{"number_uint32", "default"}, {"number_int32", "default"}};
	}

	CLASS_FILTER(PVPlottingFilterPort)
};
}

#endif
