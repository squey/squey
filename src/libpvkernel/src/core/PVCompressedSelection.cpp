/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <pvkernel/core/PVCompressedSelection.h>
#include <pvkernel/core/PVSelBitField.h>

#include <cassert>
#include <cstring>

namespace
{

/* What the first byte of the stream says the rest of it is.
 */
enum Kind : uint8_t {
	Runs = 0,     //!< tagged runs and blocks of whole words
	SetRows = 1,  //!< the gaps between the rows that are set
	ClearRows = 2 //!< the gaps between the rows that are not
};

/* A run of one word costs as much as leaving it in a block and saves nothing,
 * so runs start at two.
 */
constexpr size_t shortest_run = 2;

void put_varint(std::vector<uint8_t>& stream, uint64_t value)
{
	while (value >= 0x80) {
		stream.push_back(uint8_t(value) | 0x80);
		value >>= 7;
	}
	stream.push_back(uint8_t(value));
}

uint64_t take_varint(std::vector<uint8_t> const& stream, size_t& at)
{
	uint64_t value = 0;
	unsigned shift = 0;
	while ((stream[at] & 0x80) != 0) {
		value |= uint64_t(stream[at++] & 0x7f) << shift;
		shift += 7;
	}
	return value | (uint64_t(stream[at++]) << shift);
}

void put_word(std::vector<uint8_t>& stream, uint64_t word)
{
	const size_t at = stream.size();
	stream.resize(at + sizeof(word));
	std::memcpy(&stream[at], &word, sizeof(word));
}

uint64_t take_word(std::vector<uint8_t> const& stream, size_t& at)
{
	uint64_t word = 0;
	std::memcpy(&word, &stream[at], sizeof(word));
	at += sizeof(word);
	return word;
}

/* The runs form: equal words folded, the rest kept as they are.
 */
std::vector<uint8_t> as_runs(PVCore::PVSelBitField const& selection)
{
	const size_t words = selection.chunk_count();
	const uint64_t* const data = selection.get_buffer();

	std::vector<uint8_t> stream;
	stream.reserve(words + 16);
	stream.push_back(Runs);

	size_t i = 0;
	while (i < words) {
		size_t run = 1;
		while (i + run < words && data[i + run] == data[i]) {
			run++;
		}

		if (run >= shortest_run) {
			put_varint(stream, (uint64_t(run) << 1) | 1);
			put_word(stream, data[i]);
			i += run;
			continue;
		}

		/* Whatever does not repeat is kept as it is, in one block, so that a
		 * selection with no runs at all costs a handful of bytes more than
		 * itself rather than one tag per word.
		 */
		const size_t block_start = i;
		while (i < words) {
			size_t next = 1;
			while (i + next < words && data[i + next] == data[i]) {
				next++;
			}
			if (next >= shortest_run) {
				break;
			}
			i++;
		}

		put_varint(stream, uint64_t(i - block_start) << 1);
		const size_t at = stream.size();
		stream.resize(at + (i - block_start) * sizeof(uint64_t));
		std::memcpy(&stream[at], data + block_start, (i - block_start) * sizeof(uint64_t));
	}

	return stream;
}

/* The rows form: each row named by the gap from the one before it, which is
 * what makes a scattered selection cheap -- one row in a hundred leaves gaps of
 * about a hundred, and a hundred fits in one byte.
 *
 * The last word of the selection is written out as it is, because the bits past
 * the last row are written as freely as the others and naming them as rows
 * would name rows that do not exist.
 */
std::vector<uint8_t> as_rows(PVCore::PVSelBitField const& selection, bool wanted, size_t count)
{
	const size_t words = selection.chunk_count();
	const uint64_t* const data = selection.get_buffer();
	const PVRow rows = selection.count();

	std::vector<uint8_t> stream;
	stream.reserve(count + 16);
	stream.push_back(wanted ? SetRows : ClearRows);
	put_varint(stream, count);

	uint64_t previous = 0;
	for (size_t w = 0; w < words; w++) {
		uint64_t bits = wanted ? data[w] : ~data[w];

		const size_t first = w * 64;
		if (first + 64 > rows) {
			const size_t valid = rows - first;
			bits &= valid >= 64 ? ~uint64_t(0) : ((uint64_t(1) << valid) - 1);
		}

		while (bits != 0) {
			const uint64_t row = first + uint64_t(__builtin_ctzll(bits));
			bits &= bits - 1;

			put_varint(stream, row - previous);
			previous = row;
		}
	}

	put_word(stream, data[words - 1]);

	return stream;
}

} // namespace

PVCore::PVCompressedSelection PVCore::PVCompressedSelection::from(PVSelBitField const& selection)
{
	const size_t raw = selection.chunk_count() * sizeof(uint64_t);

	std::vector<uint8_t> best = as_runs(selection);

	/* Naming the rows is only tried when it could win: every row costs at
	 * least one byte, so a selection whose runs already say it in less than
	 * there are rows to name has nothing to gain, and walking them would be
	 * the slow part of folding a stack of steps.
	 */
	const size_t set = selection.bit_count();
	const size_t clear = selection.count() - set;
	const bool fewer_set = set <= clear;
	const size_t named = fewer_set ? set : clear;

	if (named + 16 < best.size()) {
		std::vector<uint8_t> rows = as_rows(selection, fewer_set, named);
		if (rows.size() < best.size()) {
			best = std::move(rows);
		}
	}

	/* Larger than what it stands for is not worth keeping: the caller is told
	 * so, and holds on to the selection itself.
	 */
	if (best.size() >= raw) {
		return PVCompressedSelection();
	}

	PVCompressedSelection compressed;
	compressed._row_count = selection.count();
	compressed._stream = std::move(best);

	return compressed;
}

PVCore::PVSelBitField PVCore::PVCompressedSelection::expand() const
{
	assert(not is_empty());

	PVSelBitField selection(_row_count);

	size_t at = 1;
	const uint8_t kind = _stream[0];

	if (kind != Runs) {
		const size_t count = size_t(take_varint(_stream, at));

		if (kind == SetRows) {
			selection.select_none();
		} else {
			selection.select_all();
		}

		uint64_t row = 0;
		for (size_t n = 0; n < count; n++) {
			row += take_varint(_stream, at);
			selection.set_line(PVRow(row), kind == SetRows);
		}

		selection.get_buffer()[selection.chunk_count() - 1] = take_word(_stream, at);

		return selection;
	}

	uint64_t* out = selection.get_buffer();

	while (at < _stream.size()) {
		const uint64_t tag = take_varint(_stream, at);
		const size_t count = size_t(tag >> 1);

		if ((tag & 1) != 0) {
			const uint64_t word = take_word(_stream, at);
			for (size_t n = 0; n < count; n++) {
				*out++ = word;
			}
		} else {
			std::memcpy(out, &_stream[at], count * sizeof(uint64_t));
			out += count;
			at += count * sizeof(uint64_t);
		}
	}

	return selection;
}
