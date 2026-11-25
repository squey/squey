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

// The columns and the rows that the description of JSON files chooses, as their import reads
// them: the condition can read a column that is not imported, the one naming the files included,
// and the rows it keeps fill chunks, nulls and all. An investigation reads them so again once
// the NRAW is gone.

#include "common.h"

#include "../../libpvkernel/plugins/common/json/PVJsonFileDescription.h"
#include "../../libpvkernel/plugins/common/json/PVJsonSchema.h"
#include "../../libpvkernel/plugins/common/parquet/PVParquetAPI.h"

#include <squey/PVRoot.h>
#include <squey/PVSource.h>

#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVSourceCreator.h>

#include <QDir>
#include <QFile>

#include <memory>
#include <string>
#include <vector>

namespace
{

/**
 * Two readings of rows by the source, two thirds of which are kept: they fill a chunk, and leave
 * rows for a last one, on their own.
 */
constexpr const int64_t EVENT_COUNT = 2 * 64 * 1024;

/** The n of the event kept in row @p row, as those of n % 3 == 2 are left out. */
int64_t n_of(PVRow row)
{
	return 3 * (int64_t(row) / 2) + int64_t(row) % 2;
}

void write(const QString& path, const QByteArray& content)
{
	QFile file(path);
	PV_ASSERT_VALID(file.open(QIODevice::WriteOnly), "file", path.toStdString());
	file.write(content);
}

Squey::PVSource& import(pvtest::TestEnv& env,
                        const std::shared_ptr<PVRush::PVJsonFileDescription>& input_desc)
{
	input_desc->set_schema(PVRush::PVJsonSchema::infer(input_desc->paths()));
	std::vector<size_t> column_indexes;
	const PVRush::PVFormat format(
	    PVRush::PVParquetAPI::format_of(
	        input_desc->schema()->with_columns(input_desc->columns())->arrow_schema(),
	        input_desc->multi_inputs(), column_indexes)
	        .documentElement());
	PVRush::PVInputType::list_inputs inputs;
	inputs << input_desc;
	return env.add_source(
	    inputs, LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("json"), format);
}

/** The events kept, with their n, and their text, but for those of a multiple of 5. */
void check_kept(const PVRush::PVNraw& nraw)
{
	PV_VALID(size_t(nraw.column_count()), size_t(2));
	PV_VALID(int64_t(nraw.row_count()), EVENT_COUNT - EVENT_COUNT / 3);
	const pvcop::db::array& numbers = nraw.column(PVCol(0));
	const pvcop::db::array& texts = nraw.column(PVCol(1));
	for (PVRow row = 0; row < nraw.row_count(); row++) {
		const int64_t n = n_of(row);
		PV_VALID(numbers.at(row), std::to_string(n), "row", row);
		const bool has_text = n % 5 != 0;
		PV_VALID(texts.is_valid(row), has_text, "row", row);
		if (has_text) {
			PV_VALID(texts.at(row), "v" + std::to_string(n), "row", row);
		}
	}
}

/** The objects of the second file whose x is above 1, after the names of the files. */
void check_named(const PVRush::PVNraw& nraw)
{
	PV_VALID(size_t(nraw.column_count()), size_t(2));
	PV_VALID(size_t(nraw.row_count()), size_t(2));
	for (PVRow row = 0; row < 2; row++) {
		PV_VALID(nraw.column(PVCol(0)).at(row), std::string("second.json"), "row", row);
		PV_VALID(nraw.column(PVCol(1)).at(row), std::to_string(row + 2), "row", row);
	}
}

} // namespace

int main()
{
	pvtest::TestEnv env;
	const QString directory = PVRush::PVNrawCacheManager::nraw_dir();
	QDir().mkpath(directory);

	QByteArray events;
	for (int64_t n = 0; n < EVENT_COUNT; n++) {
		events +=
		    R"({"n": )" + QByteArray::number(qlonglong(n)) + R"(, "kind": ")" +
		    (n % 3 == 2 ? "drop" : "keep") + "\"" +
		    (n % 5 != 0 ? R"(, "s": "v)" + QByteArray::number(qlonglong(n)) + "\"" : QByteArray()) +
		    "}\n";
	}
	const QString events_path = directory + "/events.json";
	write(events_path, events);
	auto kept_desc = std::make_shared<PVRush::PVJsonFileDescription>(QStringList{events_path});
	kept_desc->set_columns({"n", "s"});
	kept_desc->set_row_filter("kind = 'keep'");
	Squey::PVSource& kept = import(env, kept_desc);
	check_kept(kept.get_rushnraw());

	const QString first_path = directory + "/first.json";
	const QString second_path = directory + "/second.json";
	write(first_path, R"({"x": 2})"
	                  "\n");
	write(second_path, R"({"x": 1})"
	                   "\n"
	                   R"({"x": 2})"
	                   "\n"
	                   R"({"x": 3})"
	                   "\n");
	auto named_desc =
	    std::make_shared<PVRush::PVJsonFileDescription>(QStringList{first_path, second_path});
	named_desc->set_row_filter("filename = 'second.json' AND x > 1");
	Squey::PVSource& named = import(env, named_desc);
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
	check_kept(sources.front()->get_rushnraw());
	check_named(sources.back()->get_rushnraw());

	return 0;
}
