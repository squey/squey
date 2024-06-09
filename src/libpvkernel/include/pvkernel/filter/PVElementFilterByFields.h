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

#ifndef PVFILTER_PVELEMENTFILTERBYFIELDS_H
#define PVFILTER_PVELEMENTFILTERBYFIELDS_H

#include <pvkernel/filter/PVElementFilter.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <vector>
#include <algorithm>

#include "pvkernel/core/PVElement.h"
#include "pvkernel/filter/PVFilterFunction.h"

namespace PVCore
{
class PVElement;
} // namespace PVCore

namespace PVFilter
{

class PVElementFilterByFields : public PVElementFilter
{
  public:
	PVCore::PVElement& operator()(PVCore::PVElement& elt) override;
	void add_filter(PVFieldsBaseFilter_p&& f) { _ff.push_back(f); }

  protected:
	std::vector<PVFieldsBaseFilter_p> _ff;

	CLASS_FILTER_NONREG_NOPARAM(PVElementFilterByFields)
};
} // namespace PVFilter

#endif
