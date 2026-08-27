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

// A text column can hold rows the source has no value for -- a Parquet NULL is
// one -- and pvcop records those in a selection of its own rather than in the
// column. What is left in the column for such a row means nothing: arrow does
// not say which dictionary entry a null row points at.
//
// So a scan reading the column through its dictionary has to be told, or it
// emits whatever that index happens to reach, and a filter compared against it
// answers about a value the row does not hold. What is checked here is that it
// is told: a missing value comes back as NULL, like a missing value of any
// other type, and a filter answers about the rows that carry one and only
// those.
//
// The file is written here rather than kept beside the test: the format a
// Parquet source is read with is derived from its schema, so a fixture on disk
// would need that derivation to be committed alongside and kept in step.

#include "../../libpvkernel/plugins/common/parquet/PVParquetAPI.h"
#include "../../libpvkernel/plugins/common/parquet/PVParquetFileDescription.h"

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVSourceCreator.h>

#include <pvcop/db/array.h>

#include <QDir>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <cstdio>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "common.h"

// A hundred rows over four names, every fifth one missing. Few enough distinct
// values that the scan reads the column through its dictionary rather than
// building a string per row, which is the path this is about.
static constexpr size_t ROW_COUNT = 100;
// Not "PATTERN": one of the headers this pulls in already has one, and it
// only shows up where that header is the one the toolchain reaches for.
static constexpr size_t MISSING_EVERY = 5;

static std::shared_ptr<arrow::Schema> test_schema()
{
	return arrow::schema({arrow::field("name", arrow::utf8()), arrow::field("n", arrow::int64())});
}

static std::optional<std::string> name_of_row(size_t row)
{
	static const char* NAMES[] = {"alpha", "beta", "gamma", "delta"};
	if (row % MISSING_EVERY == 0) {
		return std::nullopt;
	}
	return std::string(NAMES[row % 4]);
}

static void generate_parquet_file(const std::string& path)
{
	arrow::StringBuilder names;
	arrow::Int64Builder numbers;
	for (size_t row = 0; row < ROW_COUNT; ++row) {
		const std::optional<std::string> value = name_of_row(row);
		PARQUET_THROW_NOT_OK(value ? names.Append(*value) : names.AppendNull());
		PARQUET_THROW_NOT_OK(numbers.Append(int64_t(row)));
	}

	std::shared_ptr<arrow::RecordBatch> batch = arrow::RecordBatch::Make(
	    test_schema(), ROW_COUNT, {names.Finish().ValueOrDie(), numbers.Finish().ValueOrDie()});

	std::shared_ptr<arrow::io::FileOutputStream> file =
	    arrow::io::FileOutputStream::Open(path).ValueOrDie();
	std::unique_ptr<parquet::arrow::FileWriter> writer =
	    parquet::arrow::FileWriter::Open(*test_schema(), arrow::default_memory_pool(), file)
	        .ValueOrDie();
	PARQUET_THROW_NOT_OK(writer->WriteTable(*arrow::Table::FromRecordBatches({batch}).ValueOrDie()));
	PARQUET_THROW_NOT_OK(writer->Close());
}

static size_t count_rows(const Squey::PVDuckDBQuery& query, const std::string& sql)
{
	return std::stoull(query.run_tabular(sql).rows[0][0]);
}

int main()
{
	pvtest::TestEnv env;

	// Where the import will put its own files. Nothing has been imported yet in
	// this config, so the directory is not there to be written into.
	const QString directory = PVRush::PVNrawCacheManager::nraw_dir();
	QDir().mkpath(directory);
	const std::string path = directory.toStdString() + "/null_strings.parquet";
	generate_parquet_file(path);

	// The format a parquet source is read with is derived from its schema, so
	// the inputs and the format are built here rather than sniffed from a file.
	PVRush::PVInputType::list_inputs inputs;
	auto* input_desc =
	    new PVRush::PVParquetFileDescription(QStringList{QString::fromStdString(path)});
	input_desc->disable_multi_inputs(true);
	inputs << PVRush::PVInputDescription_p(input_desc);
	PVRush::PVParquetAPI api(input_desc);
	const PVRush::PVFormat format(api.get_format().documentElement());

	Squey::PVSource& source = env.add_source(
	    inputs, LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet"), format);
	env.compute_mappings();
	env.compute_scalings();
	env.compute_views();
	std::remove(path.c_str());

	Squey::PVView* view = env.root.current_view();
	const PVRush::PVNraw& nraw = source.get_rushnraw();
	PV_VALID(size_t(nraw.row_count()), ROW_COUNT);

	const pvcop::db::array& names = nraw.column(PVCol(0));
	PV_ASSERT_VALID(names.is_string(), "the fixture's first column is not text", 0);
	// Without this the rest would pass on a column that has nothing to say.
	PV_ASSERT_VALID(names.has_invalid() != pvcop::db::NONE,
	                "the import kept no record of the missing values", 0);

	// The oracle: pvcop's own reading of the column.
	size_t missing = 0;
	std::map<std::string, size_t> present;
	for (size_t row = 0; row < ROW_COUNT; ++row) {
		if (names.is_valid(row)) {
			++present[names.at(row)];
		} else {
			++missing;
		}
	}
	PV_ASSERT_VALID(missing > 0 && missing < ROW_COUNT,
	                "the fixture is either all missing or none", missing);
	PV_ASSERT_VALID(present.size() > 1 && present.size() < ROW_COUNT,
	                "the fixture has no dictionary worth the name", present.size());

	Squey::PVDuckDBQuery query(*view);
	const std::string col = Squey::PVDuckDBQuery::quote_identifier(query.column_names()[1]);

	// --- The column is still read through its dictionary ----------------------
	// Which is the point of the slot: a missing value used to send the whole
	// column down the row-by-row path, building a string per row read where one
	// per distinct value would do.
	{
		const auto shown =
		    query.run_tabular("SELECT " + col + " FROM layers ORDER BY rowid", nullptr, ROW_COUNT);
		PV_VALID(shown.rows.size(), ROW_COUNT);
		PV_VALID(query.dictionary_columns(), size_t(1));
	}

	// --- A missing value is NULL ----------------------------------------------
	// Which is what it is for every other type, and what lets a query ask for it.
	PV_VALID(count_rows(query, "SELECT COUNT(*) FROM layers WHERE " + col + " IS NULL"), missing);
	PV_VALID(count_rows(query, "SELECT COUNT(" + col + ") FROM layers"), ROW_COUNT - missing);

	// --- And the rows that carry one read as pvcop draws them -----------------
	{
		const auto shown =
		    query.run_tabular("SELECT " + col + " FROM layers ORDER BY rowid", nullptr, ROW_COUNT);
		PV_VALID(shown.rows.size(), ROW_COUNT);
		for (size_t row = 0; row < ROW_COUNT; ++row) {
			if (names.is_valid(row)) {
				PV_VALID(shown.rows[row][0], names.at(row));
			}
		}
	}

	// --- A filter answers about those rows, and only those --------------------
	// The one that would break if the stored value of a row without one were
	// taken at face value: it is whatever arrow left there.
	for (const auto& [value, expected] : present) {
		PV_VALID(count_rows(query,
		                    "SELECT COUNT(*) FROM layers WHERE " + col + " = '" + value + "'"),
		         expected);
		// And pvcop is what answered it: it compares the valid rows against the
		// literals that converted and the invalid ones against those that did
		// not, so a row without a value is never matched against a value. The
		// answer alone would not say which path it came down.
		PV_VALID(query.pvcop_filters(), size_t(1));
	}

	// --- Text mode still shows what the listing draws -------------------------
	{
		const auto as_text = query.run_tabular(
		    "SELECT " + col + " FROM layers(text := true) ORDER BY rowid", nullptr, ROW_COUNT);
		PV_VALID(as_text.rows.size(), ROW_COUNT);
		for (size_t row = 0; row < ROW_COUNT; ++row) {
			PV_VALID(as_text.rows[row][0], names.at(row));
		}
	}

	return 0;
}
