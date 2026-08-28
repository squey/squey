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

// The sample shipped with the application is the first thing a new install can
// open, and the only thing an evaluator with no data of their own has to go on.
// It is imported here the way pressing the button imports it: the format comes
// from the file, nothing is described by hand. A sample that stopped opening
// would otherwise only be found by someone downloading a release.

#include <memory>

#include "../../plugins/common/parquet/PVParquetAPI.h"
#include "../../plugins/common/parquet/PVParquetFileDescription.h"
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <QString>
#include <QStringList>

static constexpr const size_t ROW_COUNT = 150000;
static constexpr const size_t COLUMN_COUNT = 14;

UNICODE_MAIN()
{
	if (argc < 2) {
		// Not argv[0]: under wmain() it is a wchar_t* that std::cerr will not take.
		std::cerr << "Usage: Trush_sample_dataset sample.parquet" << std::endl;
		return 1;
	}
#ifdef _WIN32
	std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
	const QString sample = QString::fromStdString(conv.to_bytes(argv[1]));
#else
	const QString sample = QString::fromStdString(std::string(argv[1]));
#endif

	pvtest::init_ctxt();

	QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
	PVRush::PVNraw nraw;

	PVRush::PVSourceCreator_p sc =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet");
	PVRush::PVNrawOutput output(nraw);
	auto* input_desc = new PVRush::PVParquetFileDescription(QStringList{sample});
	list_inputs << PVRush::PVInputDescription_p(input_desc);
	input_desc->disable_multi_inputs(true);

	// The format is derived from the file's own schema; this is what lets the
	// sample open without the format wizard.
	PVRush::PVParquetAPI parquet_api(input_desc);
	PVRush::PVFormat format(parquet_api.get_format().documentElement());
	PVRush::PVExtractor extractor(format, output, sc, list_inputs);

	PVRush::PVControllerJob_p job =
	    extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
	job->wait_end();

	PV_VALID((size_t)nraw.row_count(), ROW_COUNT);
	PV_VALID((size_t)nraw.column_count(), COLUMN_COUNT);

	// The axes the views are worth looking at: a time to plot along, numbers to
	// scale, and strings to group by. Losing any of these to a type change would
	// leave the sample importable but pointless.
	PV_VALID(format.get_axes()[PVCol(0)].get_name().toStdString(), std::string("time"));
	PV_VALID(format.get_axes()[PVCol(0)].get_type().toStdString(), std::string("time"));
	PV_VALID(format.get_axes()[PVCol(1)].get_name().toStdString(), std::string("src_ip"));
	PV_VALID(format.get_axes()[PVCol(1)].get_type().toStdString(), std::string("string"));
	PV_VALID(format.get_axes()[PVCol(4)].get_name().toStdString(), std::string("dst_port"));
	PV_VALID(format.get_axes()[PVCol(4)].get_type().toStdString(), std::string("number_int32"));

	// And the stories planted in it are still there to be found.
	const pvcop::db::array& src_ip = nraw.column(PVCol(1));
	size_t scanner_rows = 0;
	for (size_t row = 0; row < ROW_COUNT; row++) {
		scanner_rows += src_ip.at(row) == "10.0.14.87";
	}
	PV_ASSERT_VALID(scanner_rows > 1000, "the port scan went missing", scanner_rows);

	return 0;
}
