/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVPLOTTINGFILTER_H
#define PVFILTER_PVPLOTTINGFILTER_H

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/filter/PVFilterFunction.h>

#include <pvcop/db/array.h>

#include <pvbase/types.h>

#include <set>

namespace Inendi
{

class PVPlottingFilter : public PVFilter::PVFilterFunctionBase<uint32_t*, pvcop::db::array const&>,
                         public PVCore::PVRegistrableClass<PVPlottingFilter>
{
  public:
	typedef std::shared_ptr<PVPlottingFilter> p_type;
	typedef PVPlottingFilter FilterT;
	// (type, mapping) on which a plotting can be apply
	using plotting_capability = std::pair<std::string, std::string>;

  public:
	virtual void
	operator()(pvcop::db::array const& mapped, pvcop::db::array const& minmax, uint32_t* dest) = 0;

	virtual QString get_human_name() const = 0;
	virtual std::set<plotting_capability> list_usable_type() const = 0;
};

typedef PVPlottingFilter::func_type PVPlottingFilter_f;
}

#endif
