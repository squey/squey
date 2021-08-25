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

#ifndef PVFILTER_PVFIELDSPLITTERDNSFQDN_H
#define PVFILTER_PVFIELDSPLITTERDNSFQDN_H

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter
{

class PVFieldSplitterDnsFqdn : public PVFieldsFilter<one_to_many>
{

  public:
	static const char* TLD1;
	static const char* TLD2;
	static const char* TLD3;
	static const char* SUBD1;
	static const char* SUBD2;
	static const char* SUBD3;
	static const char* SUBD1_REV;
	static const char* SUBD2_REV;
	static const char* SUBD3_REV;

  public:
	PVFieldSplitterDnsFqdn(
	    PVCore::PVArgumentList const& args = PVFieldSplitterDnsFqdn::default_args());

  public:
	void set_args(PVCore::PVArgumentList const& args) override;
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields& l,
	                                           PVCore::list_fields::iterator it_ins,
	                                           PVCore::PVField& field) override;

  private:
	bool _tld1;
	bool _tld2;
	bool _tld3;
	bool _subd1;
	bool _subd2;
	bool _subd3;
	bool _subd1_rev;
	bool _subd2_rev;
	bool _subd3_rev;
	bool _need_inv;

	CLASS_FILTER(PVFilter::PVFieldSplitterDnsFqdn)
};
}

#endif // PVFILTER_PVFIELDSPLITTERDNSFQDN_H
