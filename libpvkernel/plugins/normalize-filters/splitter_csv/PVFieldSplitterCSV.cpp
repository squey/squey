/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include "PVFieldSplitterCSV.h"

static const std::vector<char> common_separators{',', ' ', '\t', ';', '|'};

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

bool PVFilter::PVFieldSplitterCSV::guess(list_guess_result_t& res, PVCore::PVField const& in_field)
{
	PVCore::PVArgumentList test_args = get_default_args();
	bool ok = false;

	_fields_expected = std::numeric_limits<size_t>::max();

	for (const auto separator : common_separators) {
		PVCore::PVField own_field(in_field);
		PVCore::list_fields lf;

		own_field.deep_copy();

		test_args["sep"] = QVariant(QChar(separator));
		set_args(test_args);

		if (one_to_many(lf, lf.begin(), own_field) > 1) {
			// We have a match
			res.push_back(list_guess_result_t::value_type(test_args, lf));
			ok = true;
		}
	}

	return ok;
}

IMPL_FILTER(PVFilter::PVFieldSplitterCSV)
