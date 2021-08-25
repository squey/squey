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

#ifndef PVFILTER_PVMAPPINGFILTERDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERDEFAULT_H

#include <inendi/PVMappingFilter.h>

#include <pvkernel/rush/PVNraw.h>

#include <boost/date_time/posix_time/posix_time.hpp>

namespace Inendi
{

class PVMappingFilterDefault;

/**
 * Signed integer mapping. It keeps integer values.
 */
class PVMappingFilterDefault : public PVMappingFilter
{
  public:
	PVMappingFilterDefault();

	/**
	 * Point mapping values directly to nraw values without any copy
	 */
	pvcop::db::array operator()(PVCol const /*col*/, PVRush::PVNraw const& /*nraw*/) override
	{
		return {};
	}

	std::unordered_set<std::string> list_usable_type() const override
	{
		return {"ipv4",          "ipv6",         "mac_address",   "time",
		        "duration",      "number_float", "number_double", "number_int64",
		        "number_uint64", "number_int32", "number_uint32", "number_int16",
		        "number_uint16", "number_int8",  "number_uint8",  "string"};
	}

	/**
	 * Metainformation for this plugin.
	 */
	QString get_human_name() const override { return QString("Default"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterDefault)
};
}

#endif
