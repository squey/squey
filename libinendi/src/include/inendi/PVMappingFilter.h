/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTER_H
#define PVFILTER_PVMAPPINGFILTER_H

#include <pvkernel/filter/PVFilterFunction.h>

#include <pvkernel/core/PVRegistrableClass.h>

#include <pvbase/types.h> // for PVCol

#include <pvcop/db/array.h>
#include <pvcop/db/algo.h>

#include <QString>

#include <memory>        // for shared_ptr
#include <string>        // for string
#include <unordered_set> // for unordered_set
namespace PVRush
{
class PVNraw;
}

namespace PVCore
{
class PVField;
}

namespace PVRush
{
class PVFormat;
}

namespace Inendi
{

class PVMappingFilter : public PVFilter::PVFilterFunctionBase<pvcop::db::array, PVCol const>,
                        public PVCore::PVRegistrableClass<PVMappingFilter>
{
  public:
	using p_type = std::shared_ptr<PVMappingFilter>;

  public:
	virtual pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) = 0;

	virtual QString get_human_name() const = 0;

	/**
	 * List of all supported types for this mapping.
	 */
	virtual std::unordered_set<std::string> list_usable_type() const = 0;

	/**
	 * Define the valid range of values.
	 *
	 * It is usually the min/max value but it can be the full possible range like
	 * for time 24H
	 */
	virtual pvcop::db::array get_minmax(pvcop::db::array const& mapped) const
	{
		return pvcop::db::algo::minmax(mapped);
	}
};

typedef PVMappingFilter::func_type PVMappingFilter_f;
}

#endif
