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

#include <simde/x86/sse2.h>

#include <algorithm>
#include <bit>
#include <cstdint>
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

/**
 * The bytes of a block of 64 equal to the one @p c repeats, as the bits of a mask
 */
uint64_t equal_mask(const simde__m128i (&block)[4], simde__m128i c)
{
	uint64_t mask = 0;
	for (int i = 0; i < 4; i++) {
		const auto bits = uint16_t(simde_mm_movemask_epi8(simde_mm_cmpeq_epi8(block[i], c)));
		mask |= uint64_t(bits) << (16 * i);
	}
	return mask;
}

/**
 * Each bit of @p x xor the ones below it: the bytes a quote opens or closes the quotes of
 */
uint64_t prefix_xor(uint64_t x)
{
	x ^= x << 1;
	x ^= x << 2;
	x ^= x << 4;
	x ^= x << 8;
	x ^= x << 16;
	x ^= x << 32;
	return x;
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

const char*
PVRush::PVCSVRecords::record_end(const char* begin, const char* end, bool more_data) const
{
	using status = PVQuotedValueEnd::status;

	const char* p = begin;
	const char* lf = find(p, end, '\n');
	while (true) {
		const char* const quote = find(p, lf, _quoting.quote);
		if (quote == lf) {
			return (lf == end and more_data) ? nullptr : lf;
		}
		p = quote + 1;
		if (quote != begin and quote[-1] != _quoting.separator) {
			// a quote within a value is a mere character
			continue;
		}
		const PVQuotedValueEnd value =
		    quoted_value_end(p, end, _quoting, more_data, max_quoted_lines);
		if (value.state == status::undecided) {
			return nullptr;
		}
		if (value.state == status::closed) {
			p = value.quote + 1;
			if (p > lf) {
				lf = find(p, end, '\n');
			}
		}
	}
}

const char* PVRush::PVCSVRecords::split_blocks(const char* record,
                                               const char* end,
                                               std::vector<const char*>& ends,
                                               const char*& stop) const
{
	const simde__m128i quote = simde_mm_set1_epi8(_quoting.quote);
	const simde__m128i separator = simde_mm_set1_epi8(_quoting.separator);
	const simde__m128i lf = simde_mm_set1_epi8('\n');
	const simde__m128i cr = simde_mm_set1_epi8('\r');

	bool inside = false;         // whether a quoted value holds the beginning of the block
	bool after_delimiter = true; // whether a field may begin with the block
	size_t quoted_lines = 0;     // the line feeds the quoted values of the record hold so far

	// a block needs the byte after it, which tells whether its last quote closes a value
	for (const char* p = record; end - p > 64; p += 64) {
		simde__m128i block[4];
		for (int i = 0; i < 4; i++) {
			block[i] = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(p + 16 * i));
		}
		const uint64_t quotes = equal_mask(block, quote);
		const uint64_t lfs = equal_mask(block, lf);

		if (quotes == 0 and not inside) {
			for (uint64_t ends_mask = lfs; ends_mask != 0; ends_mask &= ends_mask - 1) {
				ends.push_back(p + std::countr_zero(ends_mask));
				record = ends.back() + 1;
			}
			quoted_lines = lfs != 0 ? 0 : quoted_lines;
			after_delimiter = p[63] == _quoting.separator or p[63] == '\n';
			continue;
		}

		const uint64_t separators = equal_mask(block, separator);
		// the bytes quotes hold, the quote closing them included, and before it
		const uint64_t inside_after = prefix_xor(quotes) ^ (inside ? ~uint64_t(0) : 0);
		const uint64_t inside_before = inside_after ^ quotes;
		const uint64_t delimiters = separators | lfs | quotes;

		// A quote opens a value at the beginning of a field, or opens it again after the
		// first of two quotes standing for one; it closes a value before a separator, a
		// line break, or the second of two quotes. Anything else is for record_end().
		const uint64_t field_beginnings = (delimiters << 1) | uint64_t(after_delimiter);
		const char next = p[64];
		const bool next_follows =
		    next == _quoting.separator or next == '\n' or next == '\r' or next == _quoting.quote;
		uint64_t closings_followed = (delimiters >> 1) | (uint64_t(next_follows) << 63);
		if ((quotes & inside_before & ~closings_followed) != 0) {
			// the carriage returns of the line breaks
			closings_followed |= equal_mask(block, cr) >> 1;
		}
		const uint64_t quoted_lfs = lfs & inside_after;
		if ((quotes & ~inside_before & ~field_beginnings) != 0 or
		    (quotes & inside_before & ~closings_followed) != 0 or
		    quoted_lines + size_t(std::popcount(quoted_lfs)) > max_quoted_lines) {
			stop = p + 64;
			return record;
		}

		const uint64_t record_lfs = lfs & ~inside_after;
		for (uint64_t ends_mask = record_lfs; ends_mask != 0; ends_mask &= ends_mask - 1) {
			ends.push_back(p + std::countr_zero(ends_mask));
			record = ends.back() + 1;
		}
		if (record_lfs != 0) {
			// the line feeds quoted after the last record ended
			const int last = 63 - std::countl_zero(record_lfs);
			quoted_lines = size_t(std::popcount(quoted_lfs >> last));
		} else {
			quoted_lines += size_t(std::popcount(quoted_lfs));
		}
		inside = (inside_after >> 63) != 0;
		after_delimiter = (delimiters >> 63) != 0;
	}
	stop = end;
	return record;
}

const char* PVRush::PVCSVRecords::split(const char* begin,
                                        const char* end,
                                        bool more_data,
                                        std::vector<const char*>& ends) const
{
	const char* record = begin;
	// record_end() reads the records beginning before, which split_blocks() could not
	const char* scalar_until = begin;
	while (record != end) {
		if (record >= scalar_until) {
			record = split_blocks(record, end, ends, scalar_until);
			if (record == end) {
				break;
			}
		}
		const char* const record_end = this->record_end(record, end, more_data);
		if (record_end == nullptr) {
			return record;
		}
		ends.push_back(record_end);
		record = record_end == end ? end : record_end + 1;
	}
	return end;
}
