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
#include <pvkernel/core/squey_assert.h>

#include <cstdio>
#include <random>

static constexpr PVRow rows = 5'000'000;
static const size_t raw_bytes = (rows / 8);

/** Folded and unfolded again has to be the very same rows. */
static void round_trips(const char* what, PVCore::PVSelBitField const& sel, bool expect_folded)
{
	const PVCore::PVCompressedSelection folded = PVCore::PVCompressedSelection::from(sel);

	PV_VALID(not folded.is_empty(), expect_folded, "what", what);
	if (folded.is_empty()) {
		return;
	}

	PV_ASSERT_VALID(folded.expand() == sel, "what", what);

	printf("%-24s %8zu bytes for %8zu (%.0fx)\n", what, folded.bytes(), raw_bytes,
	       double(raw_bytes) / double(folded.bytes()));
}

int main()
{
	// The shapes that matter: whole selections, and stretches of rows.
	{
		PVCore::PVSelBitField sel(rows);
		sel.select_all();
		round_trips("everything", sel, true);
	}
	{
		PVCore::PVSelBitField sel(rows);
		sel.select_none();
		round_trips("nothing", sel, true);
	}
	{
		PVCore::PVSelBitField sel(rows);
		sel.select_none();
		for (PVRow i = rows / 4; i < rows / 2; i++) {
			sel.set_line(i, true);
		}
		round_trips("one stretch of rows", sel, true);
	}
	{
		// A search that kept every other row: nothing repeats, so this is the
		// case the encoder has to refuse to make worse rather than the case it
		// is for.
		PVCore::PVSelBitField sel(rows);
		sel.select_none();
		for (PVRow i = 0; i < rows; i += 2) {
			sel.set_line(i, true);
		}
		const PVCore::PVCompressedSelection folded = PVCore::PVCompressedSelection::from(sel);
		PV_ASSERT_VALID(not folded.is_empty(), "why", "one word over is still worth keeping");
		PV_ASSERT_VALID(folded.bytes() < raw_bytes);
		PV_ASSERT_VALID(folded.expand() == sel);
	}
	{
		// Random rows: no runs of whole words at all. Refused, and the caller
		// keeps what it had.
		PVCore::PVSelBitField sel(rows);
		sel.select_none();
		std::mt19937_64 gen(1234);
		for (PVRow i = 0; i < rows; i++) {
			sel.set_line(i, (gen() & 1) != 0);
		}
		round_trips("random rows", sel, false);
	}
	{
		// A few rows in a hundred, which is what a search usually keeps.
		PVCore::PVSelBitField sel(rows);
		sel.select_none();
		std::mt19937_64 gen(4321);
		for (PVRow i = 0; i < rows / 100; i++) {
			sel.set_line(gen() % rows, true);
		}
		round_trips("one row in a hundred", sel, true);
	}

	return 0;
}
