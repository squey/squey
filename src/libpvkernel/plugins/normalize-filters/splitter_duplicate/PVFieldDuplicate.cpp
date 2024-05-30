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

#include "PVFieldDuplicate.h"

/******************************************************************************
 *
 * PVFilter::PVFieldDuplicate::PVFieldDuplicate
 *
 *****************************************************************************/
PVFilter::PVFieldDuplicate::PVFieldDuplicate(PVCore::PVArgumentList const& args)
    : PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER(PVFilter::PVFieldDuplicate, args);
}

void PVFilter::PVFieldDuplicate::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_n = std::max((uint32_t)args.at("n").toUInt(), (uint32_t)2);
	set_number_expected_fields(_n);
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldDuplicate)
{
	PVCore::PVArgumentList args;
	args["n"] = 2;
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldDuplicate::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldDuplicate::one_to_many(
    PVCore::list_fields& l, PVCore::list_fields::iterator it_ins, PVCore::PVField& field)
{
	PVCore::list_fields::size_type ret = 0;

	for (size_t i = 0; i < _n; i++) {
		PVCore::PVField& ins_f(*l.insert(it_ins, field));
		ins_f.allocate_new(field.size());
		memcpy(ins_f.begin(), field.begin(), field.size());
		ins_f.set_end(ins_f.begin() + field.size());
		ret++;
	}

	return ret;
}
