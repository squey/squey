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

#ifndef PVFILTER_PVELEMENTFILTERBYAXES_H
#define PVFILTER_PVELEMENTFILTERBYAXES_H

#include <pvkernel/rush/PVFormat.h> // for PVFormat, etc

#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/filter/PVElementFilterByFields.h>

#include <pvkernel/core/PVElement.h> // for PVElement

namespace PVFilter
{

class PVElementFilterByAxes : public PVElementFilterByFields
{
  public:
	using fields_mask_t = PVRush::PVFormat::fields_mask_t;

  public:
	explicit PVElementFilterByAxes(const fields_mask_t& fields_mask);

  public:
	PVCore::PVElement& operator()(PVCore::PVElement& elt) override;

  protected:
	CLASS_FILTER_NONREG_NOPARAM(PVElementFilterByAxes)

  private:
	const fields_mask_t& _fields_mask;
};

} // namespace PVFilter

#endif // PVFILTER_PVELEMENTFILTERBYAXES_H
