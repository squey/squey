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

#ifndef PVFILTER_PVFIELDCONVERTERSUBSTITUTION_H
#define PVFILTER_PVFIELDCONVERTERSUBSTITUTION_H

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

#include <unordered_map>

namespace PVFilter
{

class PVFieldConverterSubstitution : public PVFieldsConverter
{
  public:
	enum ESubstitutionModes { NONE = 0, WHOLE_FIELD = 1 << 0, SUBSTRINGS = 1 << 1 };

  public:
	PVFieldConverterSubstitution(
	    PVCore::PVArgumentList const& args = PVFieldConverterSubstitution::default_args());

  public:
	void set_args(PVCore::PVArgumentList const& args) override;
	PVCore::PVField& one_to_one(PVCore::PVField& field) override;

  private:
	size_t _modes = ESubstitutionModes::NONE;
	bool _invert_order = false;

	// whole fields mode
	std::string _default_value;
	bool _use_default_value = false;
	char _sep_char = ',';
	char _quote_char = '"';
	std::unordered_map<std::string, std::string> _key;

	// substrings mode
	std::vector<std::pair<std::string, std::string>> _substrings_map;

	CLASS_FILTER(PVFilter::PVFieldConverterSubstitution)
};
}

#endif // PVFILTER_PVFIELDCONVERTERSUBSTITUTION_H
