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

// A script reaches the console's SQL the way the console does, and reads what
// comes back the way it reads a source: the same words for column_count,
// column_name, column_type, column and valid, so a function written against one
// works on the other.
//
// And valid() is what a source was missing. column() hands back the storage, in
// which a cell the format could not read occupies its slot with an encoding
// rather than a value -- an unreadable number reads as a plain 0. Nothing said
// which was which.

#include <squey/PVPythonInterpreter.h>
#include <squey/PVRoot.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>

#include <string>

#include "common.h"

// Thirteen rows over two axes holding the same text: six readable numbers, four
// words and three empty cells. col1 is typed as a number, so its unreadable
// cells have no numeric form; col2 is text, where every cell reads as written.
const std::string filename = TEST_FOLDER "/picviz/errors_search.csv";
const std::string fileformat = TEST_FOLDER "/picviz/errors_search.csv.format";

// Written as one script rather than several: a failing assert names its line,
// and the interpreter is started once for the run whatever it is handed.
const std::string script = R"PY(
import numpy

source = squey.source(0)
rows = source.row_count()

# --- valid() tells an unreadable cell from a real value --------------------
numbers = source.column("col1")
ok = source.valid("col1")
assert ok.dtype == numpy.dtype("bool"), ok.dtype
assert ok.size == rows
missing = int((~ok).sum())
assert 0 < missing < rows, missing
# Every slot holds something of the column's own type, unreadable ones
# included -- what sits there is an encoding, and nothing about it says so.
# Which is the whole reason for the second array.
assert numbers.size == rows
assert int(ok.sum()) < rows

# A column the format read whole says so, without a single false.
assert source.valid("col2").all()

# --- A query that summarizes gives back a result ---------------------------
counted = source.query("SELECT COUNT(*) AS n, COUNT(col1) AS readable FROM layers")
assert counted.row_count() == 1
assert counted.column_count() == 2
assert counted.column_name(0) == "n"
assert counted.column_type("readable") == "BIGINT"
assert int(counted.column("n")[0]) == rows
# COUNT() skips what has no value, which is the same count valid() gives.
assert int(counted.column("readable")[0]) == rows - missing

# --- Its columns carry their own validity ----------------------------------
listed = source.query("SELECT col1 FROM layers")
assert listed.row_count() == rows
assert int((~listed.valid("col1")).sum()) == missing
assert listed.column("col1").dtype == numpy.dtype("int64")

# A real one comes back as a real one rather than as text.
averaged = source.query("SELECT AVG(col1) AS mean FROM layers")
assert averaged.column("mean").dtype == numpy.dtype("float64")

# And what has no shape of its own comes back as the text it prints as.
worded = source.query("SELECT col2 FROM layers")
assert worded.column_type(0) == "VARCHAR"
assert worded.column("col2").size == rows

# --- A query that names rows gives the rows --------------------------------
selected = source.select("SELECT rowid FROM layers WHERE col1 > 1")
assert selected.dtype == numpy.dtype("bool"), selected.dtype
assert selected.size == rows
kept = int(selected.sum())
assert 0 < kept < rows, kept

# The condition alone says the same thing, which is the form the console takes.
assert int(source.select("col1 > 1").sum()) == kept

# And it is the array insert_layer() takes, so a query becomes a layer.
source.insert_layer("Greater", selected)
assert int(source.layer("Greater").get().sum()) == kept

# --- A query naming no rows is not a table ---------------------------------
try:
    source.select("SELECT col1, col2 FROM layers")
    assert False, "a query projecting two columns cannot be a selection"
except RuntimeError as e:
    assert "exactly one column" in str(e), str(e)
)PY";

int main()
{
	pvtest::TestEnv env(filename, fileformat, 1, pvtest::ProcessUntil::View);

	Squey::PVPythonInterpreter& python = Squey::PVPythonInterpreter::get(env.root);
	python.execute_script(script, false);

	return 0;
}
