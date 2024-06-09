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

#ifndef PVFILTER_PVFIELDFILTERGREP_H
#define PVFILTER_PVFIELDFILTERGREP_H

#include <pvkernel/filter/PVFieldsFilter.h>
#include <QString>

#include "pvkernel/filter/PVFilterFunction.h"
#include "pvkernel/core/PVField.h"

namespace PVCore {
class PVArgumentList;
class PVField;
}  // namespace PVCore

namespace PVFilter
{

class PVFieldFilterGrep : public PVFieldsFilter<one_to_one>
{
  public:
	explicit PVFieldFilterGrep(
	    PVCore::PVArgumentList const& args = PVFieldFilterGrep::default_args());

  public:
	void set_args(PVCore::PVArgumentList const& args) override;

  public:
	PVCore::PVField& one_to_one(PVCore::PVField& obj) override;

  protected:
	QString _str;
	bool _inverse;

	CLASS_FILTER(PVFilter::PVFieldFilterGrep)
};
} // namespace PVFilter

#endif
