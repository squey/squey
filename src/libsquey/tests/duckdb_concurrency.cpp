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

// A scan hands DuckDB pointers into the columns and walks them from several
// threads for as long as a query runs. Adding or removing a column moves the
// column vector and unmaps storage, so the two cannot overlap -- and until this
// was written, nothing said so: the contract was a sentence in a header, held up
// only by the fact that the console blocks its window while a query runs.
//
// The contract is now a lock on the nraw, shared by readers and taken
// exclusively by the two writers. What is checked here is both halves: a reader
// does not exclude another reader, and a writer waits.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "common.h"

const std::string filename = TEST_FOLDER "/picviz/errors_search.csv";
const std::string fileformat = TEST_FOLDER "/picviz/errors_search.csv.format";

//! Long enough for a thread to reach a lock, short enough not to be felt.
constexpr auto SETTLE = std::chrono::milliseconds(200);

int main()
{
	pvtest::TestEnv env(filename, fileformat, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();
	PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const size_t row_count = nraw.row_count();
	const PVCol column_count = nraw.column_count();
	PV_ASSERT_VALID(column_count > PVCol(1), "the test needs a column to spare", column_count);

	Squey::PVDuckDBQuery query(*view);

	// --- A reader does not exclude a reader -----------------------------------
	// Held across a query, which takes the same lock: were it not a shared one,
	// this would deadlock rather than fail.
	{
		const auto held = nraw.lock_structure();

		Squey::PVSelection out(row_count);
		query.select("SELECT rowid FROM layers", out);
		PV_VALID(out.bit_count(), row_count);

		// Two queries at once, on the same source through two objects, which is
		// what a console and a script doing the same thing amount to.
		std::atomic<size_t> counted{0};
		std::thread other([&]() {
			Squey::PVDuckDBQuery elsewhere(view->get_parent<Squey::PVSource>());
			Squey::PVSelection theirs(row_count);
			elsewhere.select("SELECT rowid FROM layers", theirs);
			counted = theirs.bit_count();
		});
		other.join();
		PV_VALID(counted.load(), row_count);
	}

	// --- A writer waits -------------------------------------------------------
	// The reader's lock is held while a column is removed from another thread:
	// the removal must not happen until the lock goes.
	//
	// The first column is the one removed, not the last: what follows it has to
	// shift down, and a column vector left as it was answers for its neighbour
	// ever after. The file holds the same values in both its columns, so one of
	// its own is appended to tell them apart.
	{
		std::vector<std::string> marks(row_count);
		std::vector<std::string_view> mark_views(row_count);
		for (size_t row = 0; row < row_count; ++row) {
			marks[row] = "mark" + std::to_string(row);
			mark_views[row] = marks[row];
		}
		PV_ASSERT_VALID(nraw.append_column(mark_views), "the appended column was refused", 0);

		const PVCol appended(nraw.column_count() - 1);
		const std::string mark = nraw.at_string(0, appended);
		PV_VALID(mark, marks[0]);

		const PVCol before_removal = nraw.column_count();

		std::atomic<bool> removed{false};
		std::thread writer;

		{
			const auto held = nraw.lock_structure();

			writer = std::thread([&]() {
				nraw.delete_column(PVCol(0));
				removed = true;
			});

			std::this_thread::sleep_for(SETTLE);
			PV_ASSERT_VALID(not removed.load(), "a column was removed under a reader", 1);
			PV_VALID(nraw.column_count(), before_removal);
		}

		writer.join();
		PV_ASSERT_VALID(removed.load(), "the writer never got through", 0);
		PV_VALID(nraw.column_count(), PVCol(before_removal - 1));

		// the appended column has shifted down by one, and still reads as itself
		PV_VALID(nraw.at_string(0, PVCol(appended - 1)), mark);
	}

	return 0;
}
