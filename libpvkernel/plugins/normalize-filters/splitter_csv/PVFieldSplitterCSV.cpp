/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include "PVFieldSplitterCSV.h"

PVFilter::PVFieldSplitterCSV::PVFieldSplitterCSV(PVCore::PVArgumentList const& args)
{
	INIT_FILTER(PVFilter::PVFieldSplitterCSV, args);
}

void PVFilter::PVFieldSplitterCSV::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_sep = args.at("sep").toChar().toLatin1();
	_quote = args.at("quote").toChar().toLatin1();
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterCSV)
{
	PVCore::PVArgumentList args;
	args["sep"] = QVariant(QChar(','));
	args["quote"] = QVariant(QChar('"'));
	return args;
}

PVCore::list_fields::size_type PVFilter::PVFieldSplitterCSV::one_to_many(
    PVCore::list_fields& l, PVCore::list_fields::iterator it_ins, PVCore::PVField& field)
{
	// FIXME : We should handle double Quote as escaped quote
	PVCore::list_fields::value_type elt(field);
	assert(elt.begin() == field.begin());
	char* cstr = elt.begin();
	size_t n = 0;

	size_t i = 0;

	while (i < field.size()) {
		if (cstr[i] == _quote) {
			++i;
			// quoted value
			elt.set_begin(cstr + i);

			while (true) {
				i = std::find(cstr + i, cstr + field.size(), _quote) - cstr;

				if (i == field.size()) {
					// we have found the end of line but not a quote
					return 0;
				}

				if (cstr[i - 1] != '\\') {
					break;
				}
			}

			// a quote, adding the new element
			elt.set_end(cstr + i);
			elt.set_physical_end(cstr + i);
			l.insert(it_ins, elt);
			++n;

			// moving after the quote
			++i;

			if (i == field.size()) {
				// all-right, we reach the end of line
				return n;
			}

			if (cstr[i] != _sep) {
				// not a value separator, should have one
				return 0;
			}

			// skipping the separator
			++i;
		} else {
			// non-quoted value
			elt.set_begin(cstr + i);
			++n;

			i = std::find(cstr + i, cstr + field.size(), _sep) - cstr;

			if (i == field.size()) {
				// all-right, we reach the end of line
				elt.set_end(cstr + i);
				elt.set_physical_end(cstr + i);
				l.insert(it_ins, elt);

				return n;
			} else if (n == _fields_expected) {
				// enough elements have been extracted, the last one contain the rest of the field
				elt.set_end(field.end());
				elt.set_physical_end(field.end());
				l.insert(it_ins, elt);

				return n;
			}

			elt.set_end(cstr + i);
			elt.set_physical_end(cstr + i);
			l.insert(it_ins, elt);

			// skipping the separator
			++i;
		}
	}

	/* we reach the last but empty field
	 */
	elt.set_begin(cstr + i);
	elt.set_end(field.end());
	elt.set_physical_end(field.end());
	l.insert(it_ins, elt);
	n++;

	return n;
}

IMPL_FILTER(PVFilter::PVFieldSplitterCSV)
