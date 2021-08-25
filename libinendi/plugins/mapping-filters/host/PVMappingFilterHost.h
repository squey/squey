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

#ifndef PVFILTER_PVMAPPINGFILTERHOST_H
#define PVFILTER_PVMAPPINGFILTERHOST_H

#include <inendi/PVMappingFilter.h>

namespace Inendi
{

/**
 * Signed integer mapping. It keeps integer values.
 */
class PVMappingFilterHost : public PVMappingFilter
{
  public:
	PVMappingFilterHost();

	/**
	 * Copy NRaw values (real integers value) as mapping value.
	 */
	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override;

	std::unordered_set<std::string> list_usable_type() const override { return {"string"}; }

	/**
	 * Metainformation for this plugin.
	 */
	QString get_human_name() const override { return QString("Host"); }

	pvcop::db::array get_minmax(pvcop::db::array const&, pvcop::db::selection const&) const override
	{
		pvcop::db::array res("number_uint32", 2);
		auto res_array = res.to_core_array<uint32_t>();
		res_array[0] = 0;
		res_array[1] = std::numeric_limits<uint32_t>::max();
		return res;
	}

	CLASS_FILTER_NOPARAM(PVMappingFilterHost)
};
}

#endif
