/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/core/PVColumnIndexes.h> // for PVColumnIndexes
#include <pvkernel/core/PVExporter.h>
#include <pvkernel/core/PVSelBitField.h> // for PVSelBitField

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

const std::string PVCore::PVExporter::default_sep_char = ",";
const std::string PVCore::PVExporter::default_quote_char = "\"";

PVCore::PVExporter::PVExporter(const std::string& file_path,
                               const PVCore::PVSelBitField& sel,
                               const PVCore::PVColumnIndexes& column_indexes,
                               PVRow step_count,
                               const export_func& f,
                               const std::string& sep_char /* = default_sep_char */,
                               const std::string& quote_char, /* = default_quote_char */
                               const std::string& header      /* = std::string() */
                               )
    : _file_path(file_path)
    , _sel(sel)
    , _column_indexes(column_indexes)
    , _step_count(step_count)
    , _sep_char(sep_char)
    , _quote_char(quote_char)
    , _f(f)
    , _compressor(file_path)
{
	assert(_column_indexes.size() != 0);

	if (not header.empty()) {
		_compressor.write(header);
	}
}

void PVCore::PVExporter::export_rows(size_t start_index)
{
	const size_t thread_count = std::thread::hardware_concurrency();

	int thread_index = -1;

	tbb::parallel_pipeline(
	    thread_count /* = max_number_of_live_token */,
	    tbb::make_filter<void, std::pair<size_t, size_t>>(
	        tbb::filter::serial_in_order,
	        [&](tbb::flow_control& fc) -> std::pair<size_t, size_t> {
		        if ((size_t)++thread_index == thread_count) {
			        fc.stop();
			        return {};
		        }

		        const size_t range = _step_count / thread_count;
		        const size_t begin_index = start_index + (thread_index * range);
		        const size_t len = (size_t)thread_index == (thread_count - 1)
		                               ? _step_count - ((thread_count - 1) * range)
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

			            if (!_sel.get_line_fast(row_index)) {
				            continue;
			            }

			            content += _f(row_index, _column_indexes, _sep_char, _quote_char) + "\n";
		            }

		            return content;

		        }) &
	        tbb::make_filter<std::string, void>(
	            tbb::filter::serial_in_order, [&](const std::string& content) {
		            try {
			            _compressor.write(content);
		            } catch (const PVCore::PVStreamingCompressorError& e) {
			            throw PVExportError(e.what());
		            }
		        }));
}

void PVCore::PVExporter::cancel()
{
	_compressor.cancel();
}

void PVCore::PVExporter::wait_finished()
{
	_compressor.wait_finished();
}
