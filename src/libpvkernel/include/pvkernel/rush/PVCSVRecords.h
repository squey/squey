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

#ifndef PVRUSH_PVCSVRECORDS_H
#define PVRUSH_PVCSVRECORDS_H

#include <cstddef>
#include <limits>

namespace PVRush
{

/**
 * The separator of the values of a CSV, and the quote that may enclose one of them.
 */
struct PVCSVQuoting {
	char separator;
	char quote;
};

/**
 * Where a quoted value ends, as quoted_value_end() tells.
 */
struct PVQuotedValueEnd {
	enum class status {
		closed,     //!< quote closes the value
		not_closed, //!< no quote closes the value, whose opening quote is a mere character
		undecided   //!< the data ends before telling
	};

	status state;
	/**
	 * The quote that closes the value, or that stops it from reading as quoted when no
	 * separator follows it
	 */
	const char* quote = nullptr;
	/**
	 * Whether a backslash escapes the quotes the value holds, rather than a second quote
	 */
	bool backslash = false;
};

/**
 * Finds the quote that closes the value whose opening quote is right before @p begin.
 *
 * A quote closes the value when a separator, a line feed, a carriage return or the end of
 * the data follows it, and two quotes in a row stand for one, as RFC 4180 writes them. A
 * value that does not read so is read again with a backslash escaping its quotes, as some
 * programs write them, if a backslash comes before the quote that stopped the first reading.
 * The first reading comes first because the other one cannot close a value ending with a
 * backslash, such as a Windows directory.
 *
 * @param more_data whether data may follow @p end, which leaves undecided a value that it
 *                  does not close before
 * @param max_lines how many line feeds the value may hold before its opening quote is
 *                  deemed a mere character
 */
PVQuotedValueEnd quoted_value_end(const char* begin,
                                  const char* end,
                                  PVCSVQuoting quoting,
                                  bool more_data,
                                  size_t max_lines = std::numeric_limits<size_t>::max());

/**
 * Removes in place the escape characters of the quoted value [@p begin, @p end) that
 * quoted_value_end() closed, and returns the new end of the value.
 */
char* unescape_quoted_value(char* begin, char* end, char quote, bool backslash);

} // namespace PVRush

#endif
