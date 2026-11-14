//
// MIT License
//
// © Squey, 2026
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

#include <pvkernel/rush/PVCSVRecords.h>

#include <algorithm>
#include <cstring>

namespace
{

const char* find(const char* begin, const char* end, char c)
{
	const void* const found = std::memchr(begin, c, size_t(end - begin));
	return found != nullptr ? static_cast<const char*>(found) : end;
}

/**
 * Whether an odd number of backslashes comes right before @p p, within [@p begin, @p p)
 */
bool escaped(const char* begin, const char* p)
{
	const char* first = p;
	while (first != begin and first[-1] == '\\') {
		--first;
	}
	return (p - first) % 2 == 1;
}

struct reading {
	PVRush::PVQuotedValueEnd end;
	bool met_escaped_quote = false;
};

reading read_quoted_value(const char* begin,
                          const char* end,
                          PVRush::PVCSVQuoting quoting,
                          bool more_data,
                          size_t max_lines,
                          bool backslash)
{
	using status = PVRush::PVQuotedValueEnd::status;

	reading r{{status::not_closed}};
	size_t lines = 0;
	for (const char* p = begin;;) {
		const char* const quote = find(p, end, quoting.quote);
		if (max_lines != std::numeric_limits<size_t>::max()) {
			lines += size_t(std::count(p, quote, '\n'));
			if (lines > max_lines) {
				return r;
			}
		}
		if (quote == end) {
			r.end.state = more_data ? status::undecided : status::not_closed;
			return r;
		}
		const bool escaped_quote = escaped(begin, quote);
		r.met_escaped_quote = r.met_escaped_quote or escaped_quote;
		if (backslash and escaped_quote) {
			p = quote + 1;
			continue;
		}
		if (quote + 1 == end) {
			r.end = more_data ? PVRush::PVQuotedValueEnd{status::undecided}
			                  : PVRush::PVQuotedValueEnd{status::closed, quote, backslash};
			return r;
		}
		const char next = quote[1];
		if (next == quoting.quote) {
			p = quote + 2;
			continue;
		}
		r.end = {next == quoting.separator or next == '\n' or next == '\r' ? status::closed
		                                                                   : status::not_closed,
		         quote, backslash};
		return r;
	}
}

} // namespace

PVRush::PVQuotedValueEnd PVRush::quoted_value_end(
    const char* begin, const char* end, PVCSVQuoting quoting, bool more_data, size_t max_lines)
{
	const reading rfc = read_quoted_value(begin, end, quoting, more_data, max_lines, false);
	if (rfc.end.state == PVQuotedValueEnd::status::not_closed and rfc.met_escaped_quote) {
		return read_quoted_value(begin, end, quoting, more_data, max_lines, true).end;
	}
	return rfc.end;
}

char* PVRush::unescape_quoted_value(char* begin, char* end, char quote, bool backslash)
{
	char* w = const_cast<char*>(find(begin, end, quote));
	for (const char* r = w; r != end; ++r) {
		if (*r == quote) {
			// the run of backslashes right before the quote is written as it was read
			if (backslash and escaped(begin, w)) {
				--w;
			} else if (r + 1 != end and r[1] == quote) {
				++r;
			}
		}
		*w++ = *r;
	}
	return w;
}
