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

#ifndef PVFILTER_PVFIELDSPLITTERIP_H
#define PVFILTER_PVFIELDSPLITTERIP_H

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter
{

/**
 * Split an IP in multiple field.
 *
 * For ipv4:
 * xxx.yyy.zzz.www
 *
 * with params = 1,3 => indexes = 2, 2
 * fields are : xxx.yyy and zzz.www
 *
 * For ipv6:
 * aaaa:zzzz:eeee:rrrr:tttt:yyyy:uuuu:iiii
 *
 * with params = 0, 5, 6 => indexes = 1, 5, 1
 * fields are : aaaa, zzzz:eeee:rrrr:tttt:yyyy and uuuu
 *
 * @note incomplete ipv6 result in empty fields.
 */
class PVFieldSplitterIP : public PVFieldsSplitter
{

  public:
	// Separator between quad information for params.
	static const QString sep;

	static const constexpr char* params_ipv4 = "0,1,2";
	static const constexpr char* params_ipv6 = "0,1,2,3,4,5,6";

  public:
	PVFieldSplitterIP(PVCore::PVArgumentList const& args = PVFieldSplitterIP::default_args());

  public:
	void set_args(PVCore::PVArgumentList const& args) override;
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields& l,
	                                           PVCore::list_fields::iterator it_ins,
	                                           PVCore::PVField& field) override;

  private:
	bool _ipv6;                   //!< Wether we split on ipv6 (or ipv4)
	std::vector<size_t> _indexes; //!< Elements to keep from previous position.

	CLASS_FILTER(PVFilter::PVFieldSplitterIP)
};
}

#endif // PVFILTER_PVFIELDSPLITTERIP_H
