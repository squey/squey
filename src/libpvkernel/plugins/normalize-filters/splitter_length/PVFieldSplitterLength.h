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

#ifndef PVFILTER_PVFIELDSPLITTERLENGTH_H
#define PVFILTER_PVFIELDSPLITTERLENGTH_H

#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter
{

class PVFieldSplitterLength : public PVFieldsFilter<one_to_many>
{
  public:
	static const constexpr char* param_length = "length";
	static const constexpr char* param_from_left = "from_left";

  public:
	PVFieldSplitterLength(PVCore::PVArgumentList const& args = default_args());

  public:
	void set_args(PVCore::PVArgumentList const& args) override;

  public:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields& l,
	                                           PVCore::list_fields::iterator it_ins,
	                                           PVCore::PVField& field) override;

  protected:
	CLASS_FILTER(PVFilter::PVFieldSplitterLength)

  private:
	size_t _length = 0;
	bool _from_left = false;
};
}

#endif // PVFILTER_PVFIELDSPLITTERLENGTH_H
