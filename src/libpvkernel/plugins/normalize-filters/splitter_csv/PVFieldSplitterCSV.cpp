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

#include "PVFieldSplitterCSV.h"

static const std::vector<char> common_separators{',', ' ', '\t', ';', '|'};

// tshark's -Tfields output backslash-escapes its field values: C0 control bytes
// are emitted C-style (\a \b \t \n \v \f \r), backslashes are doubled (\\), and a
// byte equal to the field separator is emitted as <escape><separator> (escape
// table characterised empirically against tshark 4.6 -Tfields output).
//
// Only <escape><escape> and <escape><separator> are unescaped back to a literal
// byte here; C0 control sequences (\n in particular) are intentionally left as
// their 2-character escaped form. pvcop's string dictionary storage uses a raw
// '\n' byte as its own record separator (see pvcop::db::write_dict::save), so
// turning an escaped "\n" back into a literal newline would silently truncate
// the value one storage layer down. Only the separator itself needs unescaping
// to fix the column-shift bug; leaving control-char escapes alone is safe.

PVFilter::PVFieldSplitterCSV::PVFieldSplitterCSV(PVCore::PVArgumentList const& args)
{
	INIT_FILTER(PVFilter::PVFieldSplitterCSV, args);
}

void PVFilter::PVFieldSplitterCSV::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_sep = args.at("sep").toChar().toLatin1();
	_quote = args.at("quote").toChar().toLatin1();
	// Optional: absent from plain CSV formats (stays disabled), set by the PCAP
	// format to unescape tshark's -Tfields output.
	_escape = args.contains("escape") ? args.at("escape").toChar().toLatin1() : '\0';

	// FIXME : should set its expected fields count
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterCSV)
{
	PVCore::PVArgumentList args;
	args["sep"] = QVariant(QChar(','));
	args["quote"] = QVariant(QChar('"'));
	args["escape"] = QVariant(QChar('\0')); // disabled by default (plain CSV)
	return args;
}

PVCore::list_fields::size_type PVFilter::PVFieldSplitterCSV::one_to_many(
    PVCore::list_fields& l, PVCore::list_fields::iterator it_ins, PVCore::PVField& field)
{
	if (_escape != '\0') {
		return one_to_many_escaped(l, it_ins, field);
	}

	// FIXME : We should handle double Quote as escaped quote
	char* cstr = field.begin();
	char* b;
	size_t n = 0;
	size_t i = 0;

	while (i < field.size()) {
		if (cstr[i] == _quote) {
			i++;
			// quoted value
			b = cstr + i;

			while (true) {
				i = std::find(cstr + i, cstr + field.size(), _quote) - cstr;

				if (i == field.size()) {
					// we have found the end of line but not a quote
					l.emplace(it_ins, *field.elt_parent(), b, cstr + i);
					++n;
					goto eol;
				}

				if (cstr[i - 1] == '\\') {
					// an escaped quote, we continue
					++i;
					continue;
				}

				if ((i + 1) == field.size()) {
					break;
				}

				if (cstr[i + 1] == _quote) {
					/* we have found a doubled quote, moving after them
					 * to integrate them in the field
					 */
					i += 2;
					continue;
				}
				// ensure that next char is a separator
				if (cstr[i + 1] == _sep) {
					break;
				} else {
					b--; // keep quote char at the begining of the field
					i++; // and find next separator or end of line
					goto find_sep;
				}
			}

			// a quote, adding the new element
			l.emplace(it_ins, *field.elt_parent(), b, cstr + i);
			n++;

			// moving after the quote
			i++;

			if (i == field.size()) {
				// all-right, we reach the end of line
				goto eol;
			}

			if (cstr[i] != _sep) {
				// not a value separator, should have one
				goto eol;
			}

			// skipping the separator
			i++;
		} else {
			// non-quoted value
			b = cstr + i;

		find_sep:
			n++;
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
				goto eol;
			} else if (n == _fields_expected) {
				// enough elements have been extracted, the last one contain the rest of the field
				l.emplace(it_ins, *field.elt_parent(), b, field.end());
				return n;
			}

			l.emplace(it_ins, *field.elt_parent(), b, cstr + i);

			// skipping the separator
			i++;
		}
	}

	/* we reach the last but empty field
	 */
	l.emplace(it_ins, *field.elt_parent(), cstr + i, field.end());
	++n;

eol:
	if (_fields_expected < std::numeric_limits<size_t>::max()) {
		for (; n < _fields_expected; ++n) {
			l.emplace(it_ins, *field.elt_parent());
		}
	}
	return n;
}

PVCore::list_fields::size_type PVFilter::PVFieldSplitterCSV::one_to_many_escaped(
    PVCore::list_fields& l, PVCore::list_fields::iterator it_ins, PVCore::PVField& field)
{
	char* const cstr = field.begin();
	const size_t size = field.size();
	size_t n = 0;
	size_t i = 0;

	while (i < size) {
		++n;
		// The last expected field absorbs the remainder of the line (mirrors the
		// plain-CSV path), so stop splitting on separators once we reach it.
		const bool last_expected = (n == _fields_expected);

		char* const b = cstr + i; // start of the field
		char* w = b;              // in-place write cursor for the unescaped value

		while (i < size and (last_expected or cstr[i] != _sep)) {
			if (cstr[i] == _escape and i + 1 < size and
			    (cstr[i + 1] == _sep or cstr[i + 1] == _escape)) {
				// an escaped separator is data, not a boundary; a doubled escape
				// character unescapes to one literal escape character. Any other
				// backslash sequence (e.g. "\n") is left as-is (see comment above).
				*w++ = cstr[i + 1];
				i += 2;
			} else {
				*w++ = cstr[i++];
			}
		}

		l.emplace(it_ins, *field.elt_parent(), b, w);

		if (last_expected or i == size) {
			goto eol;
		}
		++i; // skip the separator
		if (i == size) {
			// the line ends with a separator: trailing empty field
			l.emplace(it_ins, *field.elt_parent(), cstr + i, cstr + i);
			++n;
			goto eol;
		}
	}

eol:
	if (_fields_expected < std::numeric_limits<size_t>::max()) {
		for (; n < _fields_expected; ++n) {
			l.emplace(it_ins, *field.elt_parent());
		}
	}
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
