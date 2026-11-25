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
 * Check the parquet export of the NRAW of a source of any type, here a CSV file.
 *
 * Numbers, times and durations keep their types, without a time zone for times. The other values
 * come out as their text, which keeps that of the invalid values, whereas the invalid numbers
 * come out as nulls. The columns chosen come out in their order, once each, and columns of the
 * same name get a number. The file reads back in Squey.
 */

#include "../../plugins/common/parquet/PVParquetAPI.h"
#include <pvkernel/rush/PVFileListDescription.h>
#include "../../plugins/input-types/parquet/PVParquetNrawExporter.h"
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>

#include <parquet/arrow/reader.h>

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <memory>
#include <string>
#include <vector>

namespace
{

constexpr const char* CSV = "1,alpha,1.5,2024-01-02 03:04:05,2024-01-02 03:04:05.250000,10.0.0.1,"
                            "01:02:03,first\n"
                            "2,beta,oops,2024-01-03 03:04:05,2024-01-03 03:04:05.5,10.0.0.2,"
                            "00:00:10,second\n"
                            "3,gamma,3.5,2024-01-04 03:04:05,2024-01-04 03:04:05.0,10.0.0.3,"
                            "00:01:00,third\n"
                            "4,delta,-4.5,2024-01-05 03:04:05,2024-01-05 03:04:05.75,10.0.0.4,"
                            "10:00:00,fourth\n";

constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field><axis name="id" type="number_uint32"/></field>
  <field><axis name="name" type="string"/></field>
  <field><axis name="amount" type="number_double"/></field>
  <field><axis name="when" type="time" type_format="yyyy-MM-dd HH:mm:ss"/></field>
  <field><axis name="when_us" type="time" type_format="yyyy-MM-dd HH:mm:ss.S"/></field>
  <field><axis name="ip" type="ipv4"/></field>
  <field><axis name="elapsed" type="duration"/></field>
  <field><axis name="name" type="string"/></field>
 </splitter>
</param>
)";

void write(const QString& path, const QByteArray& content)
{
	QFile file(path);
	PV_ASSERT_VALID(file.open(QIODevice::WriteOnly), "file", path.toStdString());
	file.write(content);
}

/**
 * Exports @p rows of the @p columns of @p nraw to @p path, and reads the file written back.
 */
std::shared_ptr<arrow::Table> export_rows(const PVRush::PVNraw& nraw,
                                          const QStringList& names,
                                          const PVCore::PVColumnIndexes& columns,
                                          const std::vector<PVRow>& rows,
                                          const std::string& path)
{
	PVCore::PVSelBitField selection(nraw.row_count());
	selection.select_none();
	for (PVRow row : rows) {
		selection.set_bit_fast(row);
	}

	PVRush::PVParquetNrawExporter exporter(nraw, names);
	exporter.set_columns(columns);
	exporter.export_rows(path, selection);

	std::unique_ptr<parquet::arrow::FileReader> reader =
	    parquet::arrow::OpenFile(arrow::io::ReadableFile::Open(path).ValueOrDie(),
	                             arrow::default_memory_pool())
	        .ValueOrDie();
	return reader->ReadTable().ValueOrDie()->CombineChunks().ValueOrDie();
}

std::string names_of(const std::shared_ptr<arrow::Table>& table)
{
	std::string names;
	for (const std::shared_ptr<arrow::Field>& field : table->schema()->fields()) {
		names += (names.empty() ? "" : ",") + field->name() + ":" + field->type()->ToString();
	}
	return names;
}

void import_parquet(const std::string& path, PVRush::PVNraw& nraw)
{
	PVRush::PVNrawOutput output(nraw);
	auto* input_desc = new PVRush::PVFileListDescription({QString::fromStdString(path)});
	PVRush::PVInputType::list_inputs inputs;
	inputs << PVRush::PVInputDescription_p(input_desc);
	PVRush::PVParquetAPI api(input_desc);
	PVRush::PVExtractor extractor(
	    PVRush::PVFormat(api.get_format().documentElement()), output,
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet"), inputs);
	extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT)->wait_end();
}

template <class Array>
std::shared_ptr<Array> column_of(const std::shared_ptr<arrow::Table>& table, int column)
{
	return std::static_pointer_cast<Array>(table->column(column)->chunk(0));
}

} // namespace

UNICODE_MAIN()
{
	(void)argc;
	(void)argv;

	QTemporaryDir temporary_dir(QDir::tempPath() + "/export_nraw_XXXXXX");
	PV_ASSERT_VALID(temporary_dir.isValid());
	const QDir root(temporary_dir.path());
	const QString csv_path = root.filePath("values.csv");
	write(csv_path, CSV);
	write(csv_path + ".format", FORMAT);

	pvtest::TestEnv env(csv_path.toStdString(), (csv_path + ".format").toStdString());
	env.load_data();
	const PVRush::PVNraw& nraw = env._nraw;
	PV_VALID(size_t(nraw.row_count()), size_t(4));
	PV_VALID(nraw.column(PVCol(4)).type(), std::string("datetime_us"));

	QStringList names;
	for (const PVRush::PVAxisFormat& axis : env._format.get_axes()) {
		names << axis.get_name();
	}

	const std::string path = root.filePath("exported.parquet").toStdString();
	const std::shared_ptr<arrow::Table> all = export_rows(nraw, names, {}, {0, 1, 3}, path);
	PV_VALID(names_of(all),
	         std::string("id:uint32,name:string,amount:double,when:timestamp[ms],"
	                     "when_us:timestamp[us],ip:string,elapsed:duration[us],name_2:string"));
	PV_VALID(all->num_rows(), int64_t(3));

	const auto ids = column_of<arrow::UInt32Array>(all, 0);
	PV_VALID(ids->Value(0), uint32_t(1));
	PV_VALID(ids->Value(1), uint32_t(2));
	PV_VALID(ids->Value(2), uint32_t(4));
	PV_VALID(column_of<arrow::StringArray>(all, 1)->GetString(2), std::string("delta"));

	const auto amounts = column_of<arrow::DoubleArray>(all, 2);
	PV_VALID(amounts->Value(0), 1.5);
	PV_ASSERT_VALID(amounts->IsNull(1));
	PV_VALID(amounts->Value(2), -4.5);

	// 2024-01-02 03:04:05 UTC
	const int64_t seconds = 1704164645;
	PV_VALID(column_of<arrow::TimestampArray>(all, 3)->Value(0), seconds * 1000);
	PV_VALID(column_of<arrow::TimestampArray>(all, 4)->Value(0), seconds * 1000000 + 250000);
	PV_VALID(column_of<arrow::TimestampArray>(all, 4)->Value(2),
	         (seconds + 3 * 86400) * 1000000 + 750000);
	PV_VALID(column_of<arrow::StringArray>(all, 5)->GetString(1), std::string("10.0.0.2"));
	PV_VALID(column_of<arrow::DurationArray>(all, 6)->Value(0), int64_t(3723) * 1000000);
	PV_VALID(column_of<arrow::StringArray>(all, 7)->GetString(0), std::string("first"));

	const std::shared_ptr<arrow::Table> chosen =
	    export_rows(nraw, names, {PVCol(7), PVCol(2), PVCol(7), PVCol(1)}, {1}, path);
	PV_VALID(names_of(chosen), std::string("name:string,amount:double,name_2:string"));
	PV_VALID(column_of<arrow::StringArray>(chosen, 0)->GetString(0), std::string("second"));
	PV_VALID(column_of<arrow::StringArray>(chosen, 2)->GetString(0), std::string("beta"));

	// Squey reads the file back with the same values, but for the invalid ones, which are nulls
	// now, and the times, whose text differs.
	export_rows(nraw, names, {}, {0, 1, 2, 3}, path);
	PVRush::PVNraw imported;
	import_parquet(path, imported);
	PV_VALID(size_t(imported.row_count()), size_t(4));
	for (PVRow row = 0; row < 4; row++) {
		for (int column : {0, 1, 2, 5, 6, 7}) {
			const std::string expected =
			    row == 1 and column == 2 ? "" : nraw.column(PVCol(column)).at(row);
			PV_VALID(imported.column(PVCol(column)).at(row), expected, "row", row, "column",
			         column);
		}
	}

	return 0;
}
