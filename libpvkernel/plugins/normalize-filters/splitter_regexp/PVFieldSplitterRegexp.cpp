//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <QRegExp>

#include "PVFieldSplitterRegexp.h"
#include <pvkernel/core/PVBufferSlice.h>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexp::PVFieldSplitterRegexp
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterRegexp::PVFieldSplitterRegexp(PVCore::PVArgumentList const& args)
    : PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER(PVFilter::PVFieldSplitterRegexp, args);
}

PVFilter::PVFieldSplitterRegexp::PVFieldSplitterRegexp(const PVFieldSplitterRegexp& src)
    : PVFieldsFilter<PVFilter::one_to_many>(src)
{
	_regexp = src._regexp;
	_full_line = src._full_line;
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterRegexp)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterRegexp)
{
	PVCore::PVArgumentList args;
	args["regexp"] = PVCore::PVArgument(QString(""));
	args["full-line"] = PVCore::PVArgument(true);
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexp::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterRegexp::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);

	_full_line = args.at("full-line").toBool();

	QString s = args.at("regexp").toString();
	try {
		_regexp.assign(s.toStdString());
	} catch (const std::regex_error& e) {
		throw PVFilter::PVFieldsFilterInvalidArguments(
		    (std::string("Invalid regex : '") + s.toStdString() + "'").c_str());
	}

	set_number_expected_fields(QRegExp(args.at("regexp").toString()).captureCount());
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterRegexp::one_to_many
 *
 *****************************************************************************/

PVCore::list_fields::size_type PVFilter::PVFieldSplitterRegexp::one_to_many(
    PVCore::list_fields& l, PVCore::list_fields::iterator it_ins, PVCore::PVField& field)
{
	std::cmatch base_match;
	bool parse_success = true;
	if (_full_line) {
		parse_success =
		    std::regex_match<const char*>(field.begin(), field.end(), base_match, _regexp);
	} else {
		parse_success =
		    std::regex_search<const char*>(field.begin(), field.end(), base_match, _regexp);
	}

	if (parse_success) {
		for (auto it = ++base_match.begin(); it != base_match.end(); it++) {
			PVCore::list_fields::value_type elt(field);
			elt.set_begin(field.begin() +
			              std::distance(static_cast<const char*>(field.begin()), it->first));
			elt.set_end(field.begin() +
			            std::distance(static_cast<const char*>(field.begin()), it->second));
			elt.set_physical_end(
			    field.begin() + std::distance(static_cast<const char*>(field.begin()), it->second));
			l.insert(it_ins, elt);
		}

		return base_match.size() - 1;
	}

	return 0;
}
