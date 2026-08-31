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

#ifndef PVCORE_PVCOMPRESSEDSELECTION_H
#define PVCORE_PVCOMPRESSEDSELECTION_H

#include <pvbase/types.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace PVCore
{

class PVSelBitField;

/**
 * \class PVCompressedSelection
 *
 * A selection folded down to the runs it is made of, for keeping one around
 * without keeping its bit per row.
 *
 * What selections look like here is what makes this worth doing. Selecting
 * everything, selecting nothing and picking a stretch of rows in the listing
 * all come out as a handful of runs, whatever the collection weighs: an undo
 * step that selected every row costs sixteen bytes rather than twelve megabytes
 * on a hundred million of them. A search over unsorted data, on the other hand,
 * compresses to nothing at all -- which is why the encoder gives up and says so
 * rather than storing something larger than what it was given.
 *
 * The encoding walks the sixty-four bit words the selection is made of. Equal
 * words in a row become a run; the rest are copied as they are, in blocks. It
 * is deliberately the simplest thing that catches those shapes: a real
 * compressor would do better on the middle ground, and would be a dependency.
 */
class PVCompressedSelection
{
  public:
	PVCompressedSelection() = default;

  public:
	/**
	 * Folds a selection down, or gives back nothing when folding it would take
	 * more room than it already does.
	 */
	static PVCompressedSelection from(PVSelBitField const& selection);

	/**
	 * Whether anything was kept. An empty one stands for "not worth it", not
	 * for "an empty selection".
	 */
	bool is_empty() const { return _stream.empty(); }

	/**
	 * The selection back, bit for bit.
	 */
	PVSelBitField expand() const;

	/**
	 * What this occupies, to be weighed against what it stands for.
	 */
	size_t bytes() const { return _stream.size() * sizeof(uint64_t); }

  private:
	/* Tagged items over the words of the selection: (count << 1) | is_run,
	 * followed by the repeated word for a run, or by count words for a block
	 * of them.
	 */
	std::vector<uint64_t> _stream;
	PVRow _row_count = 0;
};
} // namespace PVCore

#endif /* PVCORE_PVCOMPRESSEDSELECTION_H */
