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
#include <vector>

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
		not_closed, //!< no quote closes the value, whose opening quote is a mere
		            //!< character
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

/**
 * Splits the text of a CSV into records: at the line feeds no quoted value holds.
 *
 * A value is quoted when its field begins with the quote. It holds at most max_quoted_lines
 * line feeds: an opening quote that nothing closes within as many lines is a mere
 * character, so that a stray one does not swallow the rest of a file.
 */
class PVCSVRecords
{
  public:
	static constexpr size_t max_quoted_lines = 1000;

  public:
	explicit PVCSVRecords(PVCSVQuoting quoting) : _quoting(quoting) {}

	PVCSVQuoting quoting() const { return _quoting; }

	/**
	 * The end of the record starting at @p begin: its line feed, or @p end when the data
	 * ends the record.
	 *
	 * @return nullptr if the data ends before telling, which @p more_data allows
	 */
	const char* record_end(const char* begin, const char* end, bool more_data) const;

	/**
	 * Appends to @p ends the end of each record of [@p begin, @p end), as record_end()
	 * tells it, and returns the beginning of the first record the data does not end, which
	 * is @p end when it ends them all.
	 */
	const char*
	split(const char* begin, const char* end, bool more_data, std::vector<const char*>& ends) const;

  private:
	/**
	 * Splits the records from @p record on, 64 bytes at a time, as long as their quotes
	 * read in bulk: as RFC 4180 writes them, within max_quoted_lines.
	 *
	 * @param[out] stop the end of the block that does not read so, or @p end
	 *
	 * @return the beginning of the record the blocks stop in
	 */
	const char* split_blocks(const char* record,
	                         const char* end,
	                         std::vector<const char*>& ends,
	                         const char*& stop) const;

  private:
	PVCSVQuoting _quoting;
};

} // namespace PVRush

#endif
