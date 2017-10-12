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

	// FIXME : should set its expected fields count
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
	char* cstr = field.begin();
	char* b;
	size_t n = 0;

	size_t i = 0;

	while (i < field.size()) {
		if (cstr[i] == _quote) {
			++i;
			// quoted value
			b = cstr + i;

			while (true) {
				i = std::find(cstr + i, cstr + field.size(), _quote) - cstr;

				if (i == field.size()) {
					// we have found the end of line but not a quote
					return 0;
				}

				if (cstr[i - 1] == '\\') {
					// an escaped quote, we continue
					++i;
					continue;
				}

				if (((i + 1) < field.size()) and (cstr[i + 1] == _quote)) {
					/* we have found a doubled quote, moving after them
					 * to integrate them in the field
					 */
					i += 2;
					continue;
				}

				// no need to search further, the final quote has been found
				break;
			}

			// a quote, adding the new element
			l.emplace(it_ins, *field.elt_parent(), b, cstr + i);
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
			b = cstr + i;
			++n;

			// check for separator character not inside quotes
			bool inside_quotes = false;
			for (; i < field.size() and (cstr[i] != _sep or inside_quotes); i++) {
				if (cstr[i] == _quote) {
					inside_quotes = not inside_quotes;
				}
			}

			if (i == field.size()) {
				// all-right, we reach the end of line
				l.emplace(it_ins, *field.elt_parent(), b, cstr + i);

				return n;
			} else if (n == _fields_expected) {
				// enough elements have been extracted, the last one contain the rest of the field
				l.emplace(it_ins, *field.elt_parent(), b, field.end());

				return n;
			}

			l.emplace(it_ins, *field.elt_parent(), b, cstr + i);

			// skipping the separator
			++i;
		}
	}

	/* we reach the last but empty field
	 */
	l.emplace(it_ins, *field.elt_parent(), cstr + i, field.end());
	n++;

	return n;
}

bool PVFilter::PVFieldSplitterCSV::guess(list_guess_result_t& res, PVCore::PVField& in_field)
{
	PVCore::PVArgumentList test_args = get_default_args();
	bool ok = false;

	_fields_expected = std::numeric_limits<size_t>::max();

	for (const auto separator : common_separators) {
		PVCore::list_fields lf;

		test_args["sep"] = QVariant(QChar(separator));
		set_args(test_args);

		if (one_to_many(lf, lf.begin(), in_field) > 1) {
			// We have a match
			res.push_back(list_guess_result_t::value_type(test_args, lf));
			ok = true;
		}
	}

	return ok;
}
