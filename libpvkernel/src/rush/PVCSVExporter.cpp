/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/rush/PVCSVExporter.h>
#include <pvkernel/core/PVColumnIndexes.h> // for PVColumnIndexes
#include <pvkernel/core/PVSelBitField.h>   // for PVSelBitField
#include <pvkernel/rush/PVNraw.h>

#include <pvbase/types.h> // for PVRow

#include <cassert>    // for assert
#include <cstddef>    // for size_t
#include <functional> // for function
#include <omp.h>      // for omp_get_thread_num
#include <ostream>    // for flush, ostream
#include <string>     // for allocator, string, etc
#include <sys/stat.h> // for mkfifo
#include <cstring>    // for std::strerror
#include <atomic>

#include <fcntl.h>
#include <sys/wait.h>

#include <tbb/pipeline.h>

#include <QLayout>
#include <QFormLayout>
#include <QCheckBox>
#include <QRadioButton>

const size_t PVRush::PVCSVExporter::STEP_COUNT = 2000;
const std::string PVRush::PVCSVExporter::default_sep_char = ",";
const std::string PVRush::PVCSVExporter::default_quote_char = "\"";

PVRush::PVCSVExporter::PVCSVExporter()
{
}

PVRush::PVCSVExporter::PVCSVExporter(PVCore::PVColumnIndexes column_indexes,
                                     PVRow total_row_count,
                                     export_func_f f,
                                     const std::string& sep_char /*= default_sep_char*/,
                                     const std::string& quote_char /*= default_quote_char*/
                                     )
    : PVRush::PVExporterBase()
    , _column_indexes(column_indexes)
    , _total_row_count(total_row_count)
    , _f(f)
    , _sep_char(sep_char)
    , _quote_char(quote_char)
{
	assert(_column_indexes.size() != 0);
}

void PVRush::PVCSVExporter::export_rows(const std::string& file_path,
                                        const PVCore::PVSelBitField& sel)
{
	PVCore::PVStreamingCompressor compressor(file_path);

	// Export header
	std::string header;
	if (_export_header) {
		compressor.write(_header);
	}

	size_t exported_row_count = 0;

	while (exported_row_count < _total_row_count) {

		if (_canceled) {
			compressor.cancel();
			std::remove(file_path.c_str());
			break;
		}

		size_t step_count = std::min(STEP_COUNT, _total_row_count - exported_row_count);

		int thread_index = -1;
		const size_t thread_count = std::thread::hardware_concurrency();
		const size_t range = step_count / thread_count;

		tbb::parallel_pipeline(
		    thread_count /* = max_number_of_live_token */,
		    tbb::make_filter<void, std::pair<size_t, size_t>>(
		        tbb::filter::serial_in_order,
		        [&](tbb::flow_control& fc) -> std::pair<size_t, size_t> {
			        if ((size_t)++thread_index == thread_count) {
				        fc.stop();
				        return {};
			        }

			        const size_t begin_index = exported_row_count + (thread_index * range);
			        const size_t len = (size_t)thread_index == (thread_count - 1)
			                               ? step_count - ((thread_count - 1) * range)
			                               : range;
			        const size_t end_index = begin_index + len;

			        return std::make_pair(begin_index, end_index);
			    }) &
		        tbb::make_filter<std::pair<size_t, size_t>, std::string>(
		            tbb::filter::parallel,
		            [&](const std::pair<size_t, size_t>& range) -> std::string {

			            const size_t begin_index = range.first;
			            const size_t end_index = range.second;

			            std::string content;
			            for (PVRow row_index = begin_index; row_index < end_index; row_index++) {

				            if (!sel.get_line_fast(row_index)) {
					            continue;
				            }

				            content +=
				                _f(row_index, _column_indexes, _sep_char, _quote_char) + "\n";
			            }

			            return content;

			        }) &
		        tbb::make_filter<std::string, void>(
		            tbb::filter::serial_in_order, [&](const std::string& content) {
			            try {
				            compressor.write(content);
			            } catch (const PVCore::PVStreamingCompressorError& e) {
				            throw PVRush::PVExportError(e.what());
			            }
			        }));

		exported_row_count += step_count;

		if (_f_progress) {
			_f_progress(exported_row_count);
		}
	}

	compressor.wait_finished();

	if (_f_progress) {
		_f_progress(_total_row_count);
	}
}
