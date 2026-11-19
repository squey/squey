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

/**
 * Check the import of files of one JSON object per line.
 *
 * The fields of the objects become columns in their order of appearance, nested ones named after
 * the fields leading to them, each with the type of its values: numbers mixing integers are
 * numbers, ISO 8601 strings are times, and arrays and mixed values are JSON text. Missing values
 * are nulls, fields always null have no column, and lines that are not objects are left out. The
 * rows of several files, with the name of their file, go through chunks that span files. A
 * description can choose the columns to import.
 */

#include "../../plugins/common/json/PVJsonFileDescription.h"
#include "../../plugins/common/json/PVJsonSchema.h"
#include "../../plugins/common/parquet/PVParquetAPI.h"
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <boost/date_time/gregorian/gregorian_types.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <memory>
#include <string>
#include <vector>

namespace
{

constexpr const char* OBJECTS =
    R"({"id": 1, "name": "alpha", "score": 1, "ok": true, "when": "2024-01-02T03:04:05Z", )"
    R"("tags": ["a", "b"], "user": {"name": "bob", "age": 30}, "mixed": 1, "nothing": null, )"
    R"("port": 80})"
    "\n"
    R"({"id": 2, "name": "beta", "score": 2.5, "ok": false, "when": "2024-01-02 03:04:05.25", )"
    R"("user": {"name": "carol"}, "mixed": "x", "extra": {"deep": {"x": 1}}, "id": 9})"
    "\n"
    "   \n"
    "not json\n"
    R"({"id": 3, "name": "gamma", "when": "2017-07-05T08:38:04.312367+0200", "user": null, )"
    R"("mixed": [1], "port": "8080"})"
    "\n"
    "[1, 2]\n"
    R"({"id": 4, "name": "delta\nline", "score": -3, "big": 18446744073709551615, )"
    R"("when": "2024-01-02T03:04:05.123456789Z"})"
    "\n";

void write(const QString& path, const QByteArray& content)
{
	QFile file(path);
	PV_ASSERT_VALID(file.open(QIODevice::WriteOnly), "file", path.toStdString());
	file.write(content);
}

std::string columns_of(const PVRush::PVJsonSchema& schema)
{
	static const char* types[] = {"boolean", "integer", "number", "time", "text"};
	std::string columns;
	for (const PVRush::PVJsonSchema::Column& column : schema.columns()) {
		columns += (columns.empty() ? "" : ",") + column.name + ":" + types[int(column.type)];
	}
	return columns;
}

/**
 * Imports @p input_desc into @p nraw.
 */
void import(std::shared_ptr<PVRush::PVJsonFileDescription> input_desc,
            const PVRush::PVJsonSchema& schema,
            PVRush::PVNraw& nraw)
{
	std::vector<size_t> column_indexes;
	PVRush::PVFormat format(PVRush::PVParquetAPI::format_of(
	                            schema.arrow_schema(), input_desc->multi_inputs(), column_indexes)
	                            .documentElement());
	PVRush::PVInputType::list_inputs inputs;
	inputs << input_desc;
	PVRush::PVNrawOutput output(nraw);
	PVRush::PVExtractor extractor(
	    format, output, LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("json"),
	    inputs);
	extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT)->wait_end();
}

boost::posix_time::ptime
time_of(int year, int month, int day, int hour, int minute, int second, int microseconds)
{
	return boost::posix_time::ptime(
	    boost::gregorian::date(year, month, day),
	    boost::posix_time::hours(hour) + boost::posix_time::minutes(minute) +
	        boost::posix_time::seconds(second) + boost::posix_time::microseconds(microseconds));
}

void check_objects(const QDir& root)
{
	const QString path = root.filePath("objects.json");
	write(path, OBJECTS);

	const std::shared_ptr<const PVRush::PVJsonSchema> schema = PVRush::PVJsonSchema::infer({path});
	PV_VALID(columns_of(*schema),
	         std::string("id:integer,name:text,score:number,ok:boolean,when:time,tags:text,"
	                     "user.name:text,user.age:integer,mixed:text,port:text,extra.deep.x:"
	                     "integer,big:number"));
	PV_VALID(schema->row_count(), size_t(4));
	PV_VALID(schema->invalid_line_count(), size_t(2));
	PV_VALID(schema->first_invalid_line(), size_t(4));

	auto input_desc = std::make_shared<PVRush::PVJsonFileDescription>(QStringList{path});
	input_desc->set_schema(schema);
	PVRush::PVNraw nraw;
	import(input_desc, *schema, nraw);
	PV_VALID(size_t(nraw.row_count()), size_t(4));
	PV_VALID(size_t(nraw.column_count()), size_t(12));

	const auto text = [&](int column, PVRow row) { return nraw.column(PVCol(column)).at(row); };
	const auto valid = [&](int column, PVRow row) {
		return nraw.column(PVCol(column)).is_valid(row);
	};
	PV_VALID(nraw.column(PVCol(0)).type(), std::string("number_int64"));
	// The first value of a field given twice.
	PV_VALID(text(0, 1), std::string("2"));
	PV_VALID(text(0, 2), std::string("3"));
	PV_VALID(text(0, 3), std::string("4"));
	PV_VALID(text(1, 3), std::string("delta\\nline"));
	PV_VALID(nraw.column(PVCol(2)).type(), std::string("number_double"));
	PV_VALID(text(2, 1), std::string("2.5"));
	PV_VALID(text(2, 3), std::string("-3"));
	PV_ASSERT_VALID(not valid(2, 2));
	PV_VALID(text(3, 0), std::string("true"));
	PV_VALID(text(3, 1), std::string("false"));
	PV_ASSERT_VALID(not valid(3, 2));

	PV_VALID(nraw.column(PVCol(4)).type(), std::string("datetime_us"));
	const auto& times = nraw.column(PVCol(4)).to_core_array<boost::posix_time::ptime>();
	PV_ASSERT_VALID(times[0] == time_of(2024, 1, 2, 3, 4, 5, 0));
	PV_ASSERT_VALID(times[1] == time_of(2024, 1, 2, 3, 4, 5, 250000));
	PV_ASSERT_VALID(times[2] == time_of(2017, 7, 5, 6, 38, 4, 312367));
	PV_ASSERT_VALID(times[3] == time_of(2024, 1, 2, 3, 4, 5, 123456));

	PV_VALID(text(5, 0), std::string(R"(["a","b"])"));
	PV_ASSERT_VALID(not valid(5, 1));
	PV_VALID(text(6, 1), std::string("carol"));
	PV_ASSERT_VALID(not valid(6, 2));
	PV_VALID(text(7, 0), std::string("30"));
	PV_ASSERT_VALID(not valid(7, 1));
	PV_VALID(text(8, 0), std::string("1"));
	PV_VALID(text(8, 1), std::string("x"));
	PV_VALID(text(8, 2), std::string("[1]"));
	PV_VALID(text(9, 0), std::string("80"));
	PV_VALID(text(9, 2), std::string("8080"));
	PV_VALID(text(10, 1), std::string("1"));
	PV_ASSERT_VALID(valid(11, 3));
	PV_ASSERT_VALID(not valid(11, 0));
}

/**
 * Rows of two files, more than a chunk holds in the first one, with nulls on both sides of the
 * end of the chunk and of that of the file.
 */
void check_files(const QDir& root)
{
	constexpr const size_t FIRST_ROW_COUNT = 70001;
	constexpr const size_t SECOND_ROW_COUNT = 3;
	QByteArray first;
	for (size_t row = 0; row < FIRST_ROW_COUNT; row++) {
		first += row % 3 == 0
		             ? QByteArray(R"({"n": )") + QByteArray::number(qulonglong(row)) + "}\n"
		             : QByteArray(R"({"n": )") + QByteArray::number(qulonglong(row)) +
		                   R"(, "s": "v)" + QByteArray::number(qulonglong(row)) + "\"}\n";
	}
	const QString first_path = root.filePath("first.json");
	const QString second_path = root.filePath("second.json");
	write(first_path, first);
	write(second_path, R"({"s": "w0", "n": 70001})"
	                   "\n"
	                   R"({"n": 70002})"
	                   "\n"
	                   R"({"n": 70003, "s": "w2"})");

	const QStringList paths{first_path, second_path};
	auto input_desc = std::make_shared<PVRush::PVJsonFileDescription>(paths);
	const std::shared_ptr<const PVRush::PVJsonSchema> schema = PVRush::PVJsonSchema::infer(paths);
	PV_VALID(columns_of(*schema), std::string("n:integer,s:text"));
	PVRush::PVNraw nraw;
	import(input_desc, *schema, nraw);

	PV_VALID(size_t(nraw.row_count()), FIRST_ROW_COUNT + SECOND_ROW_COUNT);
	PV_VALID(size_t(nraw.column_count()), size_t(3));
	for (PVRow row = 0; row < nraw.row_count(); row++) {
		const bool first_file = row < FIRST_ROW_COUNT;
		PV_VALID(nraw.column(PVCol(0)).at(row),
		         std::string(first_file ? "first.json" : "second.json"), "row", row);
		PV_VALID(nraw.column(PVCol(1)).at(row), std::to_string(row), "row", row);
		const bool has_text = first_file ? row % 3 != 0 : row != FIRST_ROW_COUNT + 1;
		PV_VALID(nraw.column(PVCol(2)).is_valid(row), has_text, "row", row);
		if (has_text) {
			const std::string expected = first_file ? "v" + std::to_string(row)
			                                        : "w" + std::to_string(row - FIRST_ROW_COUNT);
			PV_VALID(nraw.column(PVCol(2)).at(row), expected, "row", row);
		}
	}
}

/**
 * The schema of a description is the one the import uses, rather than one found again: here, it
 * leaves out a field added to the file since.
 */
void check_schema_kept(const QDir& root)
{
	const QString path = root.filePath("kept.json");
	write(path, R"({"a": 1, "b": 2})"
	            "\n");
	auto input_desc = std::make_shared<PVRush::PVJsonFileDescription>(QStringList{path});
	input_desc->set_schema(PVRush::PVJsonSchema::infer({path}));
	write(path, R"({"a": 1, "b": 2, "c": 3})"
	            "\n");

	PVRush::PVNraw nraw;
	import(input_desc, *input_desc->schema(), nraw);
	PV_VALID(size_t(nraw.column_count()), size_t(2));
	PV_VALID(nraw.column(PVCol(1)).at(0), std::string("2"));
}

/**
 * The columns a description chooses, in the order of the schema, after the names of the files.
 */
void check_columns(const QDir& root)
{
	const QString first = root.filePath("chosen_first.json");
	const QString second = root.filePath("chosen_second.json");
	write(first, R"({"id": 1, "user": {"name": "bob", "age": 30}, "port": 80})"
	             "\n");
	write(second, R"({"port": 443, "id": 2, "user": {"name": "carol"}})"
	              "\n");

	for (const QStringList& paths : {QStringList{first}, QStringList{first, second}}) {
		auto input_desc = std::make_shared<PVRush::PVJsonFileDescription>(paths);
		input_desc->set_columns({"user.name", "id", "nowhere"});
		input_desc->set_schema(PVRush::PVJsonSchema::infer(paths));
		const std::shared_ptr<const PVRush::PVJsonSchema> chosen =
		    input_desc->schema()->with_columns(input_desc->columns());
		PV_VALID(columns_of(*chosen), std::string("id:integer,user.name:text"));

		PVRush::PVNraw nraw;
		import(input_desc, *chosen, nraw);
		const int first_column = paths.size() > 1 ? 1 : 0;
		PV_VALID(size_t(nraw.column_count()), size_t(2 + first_column));
		PV_VALID(size_t(nraw.row_count()), size_t(paths.size()));
		PV_VALID(nraw.column(PVCol(first_column)).at(0), std::string("1"));
		PV_VALID(nraw.column(PVCol(first_column + 1)).at(0), std::string("bob"));
		if (paths.size() > 1) {
			PV_VALID(nraw.column(PVCol(0)).at(1), std::string("chosen_second.json"));
			PV_VALID(nraw.column(PVCol(2)).at(1), std::string("carol"));
		}
	}
}

/**
 * A byte order mark, ends of lines of two characters, no end to the last line, and a line longer
 * than the blocks the lines are read by.
 */
void check_lines(const QDir& root)
{
	const QByteArray long_value(5 * 1024 * 1024, 'x');
	const QString path = root.filePath("lines.json");
	write(path,
	      "\xEF\xBB\xBF{\"v\": \"first\"}\r\n{\"v\": \"" + long_value + "\"}\r\n{\"v\": \"last\"}");

	const std::shared_ptr<const PVRush::PVJsonSchema> schema = PVRush::PVJsonSchema::infer({path});
	PV_VALID(columns_of(*schema), std::string("v:text"));
	PV_VALID(schema->invalid_line_count(), size_t(0));

	auto input_desc = std::make_shared<PVRush::PVJsonFileDescription>(QStringList{path});
	PVRush::PVNraw nraw;
	import(input_desc, *schema, nraw);
	PV_VALID(size_t(nraw.row_count()), size_t(3));
	PV_VALID(nraw.column(PVCol(0)).at(0), std::string("first"));
	PV_VALID(nraw.column(PVCol(0)).at(1, long_value.size() + 1), long_value.toStdString());
	PV_VALID(nraw.column(PVCol(0)).at(2), std::string("last"));
}

} // namespace

UNICODE_MAIN()
{
	(void)argc;
	(void)argv;

	pvtest::init_ctxt();

	QTemporaryDir temporary_dir(PVRush::PVNrawCacheManager::nraw_dir() + "/json_import_XXXXXX");
	PV_ASSERT_VALID(temporary_dir.isValid());
	const QDir root(temporary_dir.path());

	check_objects(root);
	check_files(root);
	check_schema_kept(root);
	check_columns(root);
	check_lines(root);

	return 0;
}
