/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVPLOTTINGFILTER_H
#define PVFILTER_PVPLOTTINGFILTER_H

#include <pvkernel/filter/PVFilterFunction.h> // for PVFilterFunctionBase, etc

#include <pvkernel/core/PVRegistrableClass.h> // for PVRegistrableClass

#include <pvcop/db/array.h> // for array

#include <QString> // for QString

#include <cstdint> // for uint32_t
#include <memory>  // for shared_ptr
#include <set>     // for set
#include <string>  // for string
#include <utility> // for pair

#include <boost/date_time/posix_time/posix_time.hpp>

namespace Inendi
{

template <typename T>
inline double extract_value(const T& value)
{
	return (double)value;
}

template <>
inline double extract_value(const boost::posix_time::time_duration& value)
{
	return (double)value.total_microseconds();
}

using plotting_t = uint32_t;

static constexpr const char plotting_type[] = "number_uint32";

class PVPlottingFilter : public PVFilter::PVFilterFunctionBase<pvcop::core::array<plotting_t>&,
                                                               pvcop::db::array const&>,
                         public PVCore::PVRegistrableClass<PVPlottingFilter>
{
  public:
	using value_type = plotting_t;

	// amount of reserved space to separate valid from invalid values
	static constexpr const double INVALID_RESERVED_PERCENT_RANGE = 0.05;

  public:
	typedef std::shared_ptr<PVPlottingFilter> p_type;
	typedef PVPlottingFilter FilterT;
	// (type, mapping) on which a plotting can be apply
	using plotting_capability = std::pair<std::string, std::string>;

  public:
	virtual void operator()(pvcop::db::array const& mapped,
	                        pvcop::db::array const& minmax,
	                        const pvcop::db::selection& invalid_selection,
	                        pvcop::core::array<value_type>& dest) = 0;

	virtual QString get_human_name() const = 0;
	virtual std::set<plotting_capability> list_usable_type() const = 0;

  public:
	template <typename T>
	static std::pair<double, double> extract_minmax(const pvcop::db::array& minmax)
	{
		auto& mm = minmax.to_core_array<T>();

		if (minmax.size() == 0) {
			return std::make_pair(0.0, 0.0);
		} else {
			return std::make_pair(extract_value(mm[0]), extract_value(mm[1]));
		}
	}
};

typedef PVPlottingFilter::func_type PVPlottingFilter_f;
} // namespace Inendi

#endif
