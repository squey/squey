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

// A query runs on a thread of its own so the window stays alive, and the button
// that stops it is on another. Nothing the caller holds can be polled to that
// end -- the query is inside DuckDB, which comes back when it is done -- so the
// stop has to reach in.
//
// What is checked here is that it does, that the connection is no worse for it,
// and that a stop asked for when nothing is running stays where it was said
// rather than falling on the next query. That last one is not hypothetical: the
// flag belongs to the connection, and a cancel races the last chunk of the query
// it was meant for.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include "common.h"

const std::string filename = TEST_FOLDER "/picviz/errors_search.csv";
const std::string fileformat = TEST_FOLDER "/picviz/errors_search.csv.format";

// Long enough that it cannot finish while the test is looking, short enough that
// a run which fails to stop it still ends. range() is a core table function, so
// this asks nothing of the extensions the console has switched off.
const std::string LONG_QUERY = "SELECT COUNT(*) FROM range(200000000000)";

using clock_t_ = std::chrono::steady_clock;

int main()
{
	pvtest::TestEnv env(filename, fileformat, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();
	Squey::PVDuckDBQuery query(*view);

	const size_t row_count = view->get_row_count();
	const auto count_rows = [&]() -> size_t {
		return std::stoull(query.run_tabular("SELECT COUNT(*) FROM layers").rows[0][0]);
	};

	// --- A running query stops when asked -------------------------------------
	{
		std::atomic<bool> done{false};
		std::string error;
		std::thread worker([&]() {
			try {
				query.run_tabular(LONG_QUERY);
			} catch (const std::exception& e) {
				error = e.what();
			}
			done.store(true);
		});

		// Asked repeatedly rather than once after a sleep: a stop that arrives
		// before the query started is dropped on purpose, and this is what the
		// hand on the button does anyway.
		const auto deadline = clock_t_::now() + std::chrono::seconds(30);
		while (not done.load() && clock_t_::now() < deadline) {
			query.interrupt();
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
		}
		const bool stopped = done.load();
		worker.join();

		PV_ASSERT_VALID(stopped, "the query ran past the deadline -- it was not stopped", 0);
		PV_ASSERT_VALID(not error.empty(), "a stopped query has to fail rather than answer", 0);
	}

	// --- And the connection is no worse for it --------------------------------
	PV_VALID(count_rows(), row_count);

	// --- A stop asked for with nothing running falls nowhere ------------------
	// Said twice, since the flag is a flag: one left raised would take the very
	// next query, and it is the second one that would show a flag cleared per
	// query rather than per call.
	query.interrupt();
	query.interrupt();
	PV_VALID(count_rows(), row_count);
	PV_VALID(count_rows(), row_count);

	// --- Including for a query that yields a selection ------------------------
	{
		Squey::PVSelection selected(row_count);
		query.select("SELECT rowid FROM layers", selected);
		PV_VALID(size_t(selected.bit_count()), row_count);
	}

	return 0;
}
