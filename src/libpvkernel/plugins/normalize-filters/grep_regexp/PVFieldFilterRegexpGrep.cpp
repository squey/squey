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

#include "PVFieldFilterRegexpGrep.h"

/******************************************************************************
 *
 * PVFilter::PVFieldFilterRegexpGrepg::PVFieldFilterGrep
 *
 *****************************************************************************/
PVFilter::PVFieldFilterRegexpGrep::PVFieldFilterRegexpGrep(PVCore::PVArgumentList const& args)
    : PVFilter::PVFieldFilterGrep()
{
	INIT_FILTER(PVFilter::PVFieldFilterRegexpGrep, args);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(PVFilter::PVFieldFilterRegexpGrep)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(PVFilter::PVFieldFilterRegexpGrep)
{
	PVCore::PVArgumentList args;
	args["regexp"] = QString("");
	args["reverse"] = false;
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldFilterRegexpGrep::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldFilterRegexpGrep::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	QStringList rxs = args.at("regexp").toString().split("\n", Qt::SkipEmptyParts);
	_rxs.resize(rxs.size());
	for (int i = 0; i < rxs.size(); i++) {
		_rxs[i].assign(rxs[i].toStdString());
	}

	_inverse = args.at("reverse").toBool();
}

/******************************************************************************
 *
 * PVFilter::PVFieldFilterRegexpGrep::one_to_one
 *
 *****************************************************************************/
PVCore::PVField& PVFilter::PVFieldFilterRegexpGrep::one_to_one(PVCore::PVField& field)
{
	std::cmatch base_match;

	bool found = std::any_of(_rxs.begin(), _rxs.end(), [&](const std::regex& rx) {
		return std::regex_search<const char*>(field.begin(), field.end(), base_match, rx);
	});

	if (not(found ^ _inverse)) {
		field.set_filtered();
	}
	return field;
}
