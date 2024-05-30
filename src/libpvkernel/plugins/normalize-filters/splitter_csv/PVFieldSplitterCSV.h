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

#ifndef PVFILTER_PVFIELDSPLITTERCSV_FILE_H
#define PVFILTER_PVFIELDSPLITTERCSV_FILE_H

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

namespace PVFilter
{

class PVFieldSplitterCSV : public PVFieldsFilter<one_to_many>
{
  public:
	PVFieldSplitterCSV(PVCore::PVArgumentList const& args = PVFieldSplitterCSV::default_args());

  public:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields& l,
	                                           PVCore::list_fields::iterator it_ins,
	                                           PVCore::PVField& field) override;

	bool guess(list_guess_result_t& res, PVCore::PVField& in_field) override;

	void set_args(PVCore::PVArgumentList const& args) override;

  private:
	char _sep = ',';
	char _quote = '"';

	CLASS_FILTER(PVFilter::PVFieldSplitterCSV)
};
}

#endif
