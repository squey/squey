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

// The columns and the rows that the description of Parquet files chooses, as their import reads
// them: the condition can read a column that is not imported, a field of a structure, or the one
// naming the files, and the batches it keeps no row of are skipped. An investigation reads them so
// again once the NRAW is gone.

#include "common.h"

#include "../../libpvkernel/plugins/common/parquet/PVParquetAPI.h"
#include <pvkernel/rush/PVFileListDescription.h>

#include <squey/PVRoot.h>
#include <squey/PVSource.h>

#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVSourceCreator.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <QDir>

#include <memory>
#include <string>
#include <vector>

namespace
{

/** More rows than two batches of the reader hold, the last of which is kept in part. */
constexpr const int64_t BIG_ROW_COUNT = 150000;
constexpr const int64_t FIRST_KEPT_ID = 140000;

/**
 * Writes the rows of the ids [@p first_id, @p end_id[: their name, but for every seventh, and
 * their position, of y the last digit of the id.
 */
void write_ids(const QString& path, int64_t first_id, int64_t end_id)
{
	arrow::Int64Builder ids;
	arrow::StringBuilder names;
	arrow::DoubleBuilder xs;
	arrow::DoubleBuilder ys;
	for (int64_t id = first_id; id < end_id; id++) {
		PV_ASSERT_VALID(ids.Append(id).ok());
		PV_ASSERT_VALID(
		    (id % 7 == 0 ? names.AppendNull() : names.Append("n" + std::to_string(id))).ok());
		PV_ASSERT_VALID(xs.Append(double(id)).ok());
		PV_ASSERT_VALID(ys.Append(double(id % 10)).ok());
	}
	const arrow::FieldVector position = {arrow::field("x", arrow::float64()),
	                                     arrow::field("y", arrow::float64())};
	const std::shared_ptr<arrow::Table> table = arrow::Table::Make(
	    arrow::schema({arrow::field("id", arrow::int64()), arrow::field("name", arrow::utf8()),
	                   arrow::field("position", arrow::struct_(position))}),
	    {ids.Finish().ValueOrDie(), names.Finish().ValueOrDie(),
	     arrow::StructArray::Make({xs.Finish().ValueOrDie(), ys.Finish().ValueOrDie()}, position)
	         .ValueOrDie()});
	PV_ASSERT_VALID(parquet::arrow::WriteTable(
	                    *table, arrow::default_memory_pool(),
	                    arrow::io::FileOutputStream::Open(path.toStdString()).ValueOrDie())
	                    .ok());
}

Squey::PVSource& import(pvtest::TestEnv& env,
                        const QStringList& paths,
                        const QStringList& columns,
                        const QString& row_filter)
{
	auto* input_desc = new PVRush::PVFileListDescription(paths);
	input_desc->set_columns(columns);
	input_desc->set_row_filter(row_filter);
	PVRush::PVInputType::list_inputs inputs;
	inputs << PVRush::PVInputDescription_p(input_desc);
	PVRush::PVParquetAPI api(input_desc);
	return env.add_source(inputs,
	                      LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet"),
	                      PVRush::PVFormat(api.get_format().documentElement()));
}

/** The ids and names of the rows from FIRST_KEPT_ID on, of a position.y under 5, with a name. */
void check_big(const PVRush::PVNraw& nraw)
{
	PV_VALID(size_t(nraw.column_count()), size_t(2));
	std::vector<int64_t> ids;
	for (int64_t id = FIRST_KEPT_ID; id < BIG_ROW_COUNT; id++) {
		if (id % 10 < 5 and id % 7 != 0) {
			ids.push_back(id);
		}
	}
	PV_VALID(size_t(nraw.row_count()), ids.size());
	for (PVRow row = 0; row < nraw.row_count(); row++) {
		PV_VALID(nraw.column(PVCol(0)).at(row), std::to_string(ids[row]), "row", row);
		PV_VALID(nraw.column(PVCol(1)).at(row), "n" + std::to_string(ids[row]), "row", row);
	}
}

/**
 * The rows of the second file of an id above 11, of all of the columns, after the names of the
 * files.
 */
void check_named(const PVRush::PVNraw& nraw)
{
	PV_VALID(size_t(nraw.column_count()), size_t(5));
	PV_VALID(size_t(nraw.row_count()), size_t(3));
	for (PVRow row = 0; row < 3; row++) {
		PV_VALID(nraw.column(PVCol(0)).at(row), std::string("second.parquet"), "row", row);
		PV_VALID(nraw.column(PVCol(1)).at(row), std::to_string(12 + row), "row", row);
	}
}

} // namespace

int main()
{
	pvtest::TestEnv env;
	const QString directory = PVRush::PVNrawCacheManager::nraw_dir();
	QDir().mkpath(directory);

	const QString big = directory + "/big.parquet";
	write_ids(big, 0, BIG_ROW_COUNT);
	Squey::PVSource& big_source =
	    import(env, {big}, {"id", "name"},
	           QString(R"(id >= %1 AND "position.y" < 5 AND name IS NOT NULL)").arg(FIRST_KEPT_ID));
	check_big(big_source.get_rushnraw());

	const QString first = directory + "/first.parquet";
	const QString second = directory + "/second.parquet";
	write_ids(first, 0, 5);
	write_ids(second, 10, 15);
	Squey::PVSource& named =
	    import(env, {first, second}, {}, "filename = 'second.parquet' AND id > 11");
	check_named(named.get_rushnraw());

	env.compute_mappings();
	env.compute_scalings();
	env.compute_views();
	const std::string investigation = directory.toStdString() + "/filters.pvi";
	{
		PVCore::PVSerializeArchiveZip archive(QString::fromStdString(investigation),
		                                      PVCore::PVSerializeArchive::write,
		                                      SQUEY_ARCHIVES_VERSION, false);
		env.root.save_to_file(archive);
	}
	for (const Squey::PVSource* source : env.root.get_children<Squey::PVSource>()) {
		PVCore::PVDirectory::remove_rec(QString::fromStdString(source->get_rushnraw().dir()));
	}

	Squey::PVRoot root;
	{
		PVCore::PVSerializeArchiveZip archive(QString::fromStdString(investigation),
		                                      PVCore::PVSerializeArchive::read,
		                                      SQUEY_ARCHIVES_VERSION, false);
		root.load_from_archive(archive);
	}
	const auto sources = root.get_children<Squey::PVSource>();
	PV_VALID(sources.size(), size_t(2));
	check_big(sources.front()->get_rushnraw());
	check_named(sources.back()->get_rushnraw());

	return 0;
}
