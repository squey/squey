/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVFILTER_PVSCALINGFILTER_H
#define PVFILTER_PVSCALINGFILTER_H

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
#include <pvcop/types/datetime_us.h>

namespace Squey
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

template <>
inline double extract_value(const boost::posix_time::ptime& value)
{
	return (double)pvcop::types::formatter_datetime_us::cal(value).as_value;
}

using scaling_t = uint32_t;

static constexpr const char scaling_type[] = "number_uint32";

class PVScalingFilter : public PVFilter::PVFilterFunctionBase<pvcop::core::array<scaling_t>&,
                                                               pvcop::db::array const&>,
                         public PVCore::PVRegistrableClass<PVScalingFilter>
{
  public:
	using value_type = scaling_t;

	// amount of reserved space to separate valid from invalid values
	static constexpr const double INVALID_RESERVED_PERCENT_RANGE = 0.05;

  public:
	typedef std::shared_ptr<PVScalingFilter> p_type;
	typedef PVScalingFilter FilterT;
	// (type, mapping) on which a scaling can be apply
	using scaling_capability = std::pair<std::string, std::string>;

  public:
	virtual void operator()(pvcop::db::array const& mapped,
	                        pvcop::db::array const& minmax,
	                        const pvcop::db::selection& invalid_selection,
	                        pvcop::core::array<value_type>& dest) = 0;

	virtual QString get_human_name() const = 0;
	virtual std::set<scaling_capability> list_usable_type() const = 0;

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

typedef PVScalingFilter::func_type PVScalingFilter_f;
} // namespace Squey

#endif
