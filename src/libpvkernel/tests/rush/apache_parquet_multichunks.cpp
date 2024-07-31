/* * MIT License
 *
 * © Squey, 2024
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
#include "../../plugins/common/parquet/PVParquetAPI.h"
#include "../../plugins/common/parquet/PVParquetFileDescription.h"
#include "../../plugins/input-types/parquet/PVParquetExporter.h"
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVCSVExporter.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <pvlogger.h>


void import_files(
    const QStringList& files,
    QList<std::shared_ptr<PVRush::PVInputDescription>>& list_inputs,
    PVRush::PVFormat& format,
    PVRush::PVNraw& nraw
)
{
    PVRush::PVSourceCreator_p sc =
        LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet");
    PVRush::PVNrawOutput output(nraw);
    PVRush::PVParquetFileDescription* input_desc = new PVRush::PVParquetFileDescription(files);
    list_inputs << PVRush::PVInputDescription_p(input_desc);
    input_desc->disable_multi_inputs(true);
    PVRush::PVParquetAPI parquet_api(input_desc);
    format = PVRush::PVFormat(parquet_api.get_format().documentElement());
    PVRush::PVExtractor extractor(format, output, sc, list_inputs);

    // Import data
    auto start = std::chrono::system_clock::now();
    PVRush::PVControllerJob_p job =
        extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
    job->wait_end();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << diff.count();
}

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr
		    << "Usage: " << argv[0]
		    << "<parquet_path> <csv_ref_path>"
		    << std::endl;
		return 1;
	}

    const std::string& parquet_test_file = argv[1];
    const std::string& csv_ref_file = argv[2];

    pvtest::init_ctxt();

    // Import multichunks parquet file
    QStringList files;
    files << QString::fromStdString(parquet_test_file);
    QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
    PVRush::PVFormat format;
    PVRush::PVNraw nraw;
    import_files(files, list_inputs, format, nraw);

    // Export as CSV file and check if content is same as a csv reference file
    PVCore::PVSelBitField sel(nraw.row_count());
    sel.select_all();
    const std::string& output_csv_file = pvtest::get_tmp_filename() + ".parquet.csv";
    PVRush::PVCSVExporter::export_func_f export_func =
        [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
            const std::string& quote) { return nraw.export_line(row, cols, sep, quote); };
    PVRush::PVCSVExporter exp_csv(format.get_axes_comb(), nraw.row_count(), export_func);
    exp_csv.export_rows(output_csv_file, sel);
    std::cout << std::endl << output_csv_file << " - " << csv_ref_file << std::endl;
    PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_csv_file, csv_ref_file));

    // Cleanup files
    std::remove(output_csv_file.c_str());

	return 0;
}