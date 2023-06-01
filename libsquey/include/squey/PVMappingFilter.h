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
} // namespace PVRush

namespace PVCore
{
class PVField;
} // namespace PVCore

namespace PVRush
{
class PVFormat;
} // namespace PVRush

namespace Squey
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
	virtual pvcop::db::array get_minmax(pvcop::db::array const& mapped,
	                                    pvcop::db::selection const& valid_elts) const
	{
		return pvcop::db::algo::minmax(mapped, valid_elts);
	}
};

typedef PVMappingFilter::func_type PVMappingFilter_f;
} // namespace Squey

#endif
