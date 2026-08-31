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

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

/* A run of one word costs as much as leaving it in a block and saves nothing,
 * so runs start at two.
 */
static constexpr size_t shortest_run = 2;

namespace
{

enum Kind : uint64_t { Runs = 0, SetRows = 1, ClearRows = 2 };

/* Rows are packed two to a word; the last chunk of the selection is copied
 * verbatim beside them, so that the bits past the last row -- which the
 * operations here write as freely as the others -- come back as they were.
 */
size_t rows_form_size(size_t rows)
{
	return 3 + (rows + 1) / 2;
}

void write_rows(std::vector<uint64_t>& stream,
                PVCore::PVSelBitField const& selection,
                bool wanted,
                size_t count)
{
	stream.push_back(count);

	uint64_t packed = 0;
	size_t half = 0;
	for (PVRow row = 0; row < selection.count(); row++) {
		if (selection.get_line(row) != wanted) {
			continue;
		}
		packed |= uint64_t(row) << (32 * half);
		if (++half == 2) {
			stream.push_back(packed);
			packed = 0;
			half = 0;
		}
	}
	if (half != 0) {
		stream.push_back(packed);
	}

	stream.push_back(selection.get_buffer()[selection.chunk_count() - 1]);
}

} // namespace

PVCore::PVCompressedSelection PVCore::PVCompressedSelection::from(PVSelBitField const& selection)
{
	const size_t words = selection.chunk_count();
	const uint64_t* const data = selection.get_buffer();

	PVCompressedSelection compressed;
	compressed._row_count = selection.count();

	/* Three ways of saying the same thing, and the shortest wins. Runs suit a
	 * selection made of stretches; naming the rows suits one a search left
	 * scattered, either the few it kept or the few it dropped.
	 */
	std::vector<uint64_t> runs;
	runs.reserve(words / 8 + 4);
	runs.push_back(Runs);

	size_t i = 0;
	while (i < words) {
		size_t run = 1;
		while (i + run < words && data[i + run] == data[i]) {
			run++;
		}

		if (run >= shortest_run) {
			runs.push_back((uint64_t(run) << 1) | 1);
			runs.push_back(data[i]);
			i += run;
			continue;
		}

		/* Whatever does not repeat is kept as it is, in one block, so that a
		 * selection with no runs at all costs a single word more than itself.
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

		runs.push_back(uint64_t(i - block_start) << 1);
		runs.insert(runs.end(), data + block_start, data + i);
	}

	const size_t set = selection.bit_count();
	const size_t clear = selection.count() - set;

	const size_t best = std::min({runs.size(), rows_form_size(set), rows_form_size(clear)});

	/* Larger than what it stands for is not worth keeping: the caller is told
	 * so, and holds on to the selection itself.
	 */
	if (best >= words) {
		return PVCompressedSelection();
	}

	if (best == runs.size()) {
		compressed._stream = std::move(runs);
	} else if (best == rows_form_size(set)) {
		compressed._stream.reserve(best);
		compressed._stream.push_back(SetRows);
		write_rows(compressed._stream, selection, true, set);
	} else {
		compressed._stream.reserve(best);
		compressed._stream.push_back(ClearRows);
		write_rows(compressed._stream, selection, false, clear);
	}

	return compressed;
}

PVCore::PVSelBitField PVCore::PVCompressedSelection::expand() const
{
	assert(not is_empty());

	PVSelBitField selection(_row_count);

	const uint64_t kind = _stream[0];

	if (kind != Runs) {
		const size_t count = _stream[1];

		if (kind == SetRows) {
			selection.select_none();
		} else {
			selection.select_all();
		}

		for (size_t n = 0; n < count; n++) {
			const uint64_t packed = _stream[2 + n / 2];
			const PVRow row = PVRow((packed >> (32 * (n % 2))) & 0xffffffff);
			selection.set_line(row, kind == SetRows);
		}

		selection.get_buffer()[selection.chunk_count() - 1] = _stream.back();

		return selection;
	}

	uint64_t* out = selection.get_buffer();

	for (size_t i = 1; i < _stream.size();) {
		const uint64_t tag = _stream[i++];
		const size_t count = size_t(tag >> 1);

		if ((tag & 1) != 0) {
			const uint64_t word = _stream[i++];
			for (size_t n = 0; n < count; n++) {
				*out++ = word;
			}
		} else {
			std::memcpy(out, &_stream[i], count * sizeof(uint64_t));
			out += count;
			i += count;
		}
	}

	return selection;
}
