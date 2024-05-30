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

#include "PVFieldSplitterLength.h"

/******************************************************************************
 * PVFilter::PVFieldSplitterLength::PVFieldSplitterLength
 ******************************************************************************/

PVFilter::PVFieldSplitterLength::PVFieldSplitterLength(PVCore::PVArgumentList const& args)
{
	INIT_FILTER(PVFilter::PVFieldSplitterLength, args);
	set_number_expected_fields(2);
}

/******************************************************************************
 * PVFilter::PVFieldSplitterLength::default_args
 ******************************************************************************/

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterLength)
{
	PVCore::PVArgumentList args;

	args[param_length] = QVariant(0);
	args[param_from_left] = QVariant(true);

	return args;
}

/******************************************************************************
 * PVFilter::PVFieldSplitterLength::set_args
 ******************************************************************************/

void PVFilter::PVFieldSplitterLength::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);

	_length = std::max(0, args.at(param_length).toInt());
	_from_left = args.at(param_from_left).toBool();
}

/******************************************************************************
 * PVFilter::PVFieldSplitterLength::one_to_many
 ******************************************************************************/

PVCore::list_fields::size_type PVFilter::PVFieldSplitterLength::one_to_many(
    PVCore::list_fields& l, PVCore::list_fields::iterator it_ins, PVCore::PVField& field)
{
	PVCore::list_fields::value_type elt(field);

	if (_from_left) {
		if (field.size() <= _length) {
			// the first contains the whole original field
			l.insert(it_ins, elt);

			// the second field is empty
			elt.set_end(field.begin());
			elt.set_physical_end(field.begin());
			l.insert(it_ins, elt);
		} else {
			char* pos = field.begin() + _length;

			elt.set_end(pos);
			elt.set_physical_end(pos);
			l.insert(it_ins, elt);

			elt.set_begin(pos);
			elt.set_end(field.end());
			elt.set_physical_end(field.end());
			l.insert(it_ins, elt);
		}
	} else if (field.size() <= _length) {
		// the first field is empty
		elt.set_end(field.begin());
		elt.set_physical_end(field.begin());
		l.insert(it_ins, elt);

		// the second contains the whole original field
		elt.set_end(field.end());
		elt.set_physical_end(field.end());
		l.insert(it_ins, elt);
	} else {
		char* pos = field.end() - _length;

		elt.set_end(pos);
		elt.set_physical_end(pos);
		l.insert(it_ins, elt);

		elt.set_begin(pos);
		elt.set_end(field.end());
		elt.set_physical_end(field.end());
		l.insert(it_ins, elt);
	}

	return 2;
}
