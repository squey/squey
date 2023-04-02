//
// MIT License
//
// © ESI Group, 2015
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

#include <pvkernel/rush/PVAggregator.h>    // for PVAggregator
#include <pvkernel/rush/PVControllerJob.h> // for PVControllerJob, etc
#include <pvkernel/rush/PVExtractor.h>     // for PVExtractor
#include <pvkernel/rush/PVFormat.h>        // for PVFormat
#include <pvkernel/rush/PVInputType.h>     // for PVInputType, etc
#include <pvkernel/rush/PVNraw.h>          // for PVNraw
#include <pvkernel/rush/PVNrawOutput.h>    // for PVNrawOutput
#include <pvkernel/rush/PVSourceCreator.h> // for PVSourceCreator_p, etc
#include <pvkernel/rush/PVRawSourceBase.h>

#include <pvkernel/core/PVConfig.h> // for PVConfig

#include <pvbase/types.h> // for chunk_index, PVCol

#include <tbb/task_scheduler_init.h> // for task_scheduler_init

#include <QSettings> // for QSettings

#include <algorithm> // for min
#include <cstdint>   // for uint32_t
#include <limits>    // for numeric_limits
#include <memory>    // for __shared_ptr

PVRush::PVExtractor::PVExtractor(const PVRush::PVFormat& format,
                                 PVRush::PVOutput& output,
                                 PVRush::PVSourceCreator_p src_plugin,
                                 PVRush::PVInputType::list_inputs const& inputs)
    : _output(output)
    , _format(format)
    , _chk_flt(_format.create_tbb_filters())
    , _chunks(tbb::task_scheduler_init::default_num_threads())
    , _max_value(0)
{
	for (auto const& input : inputs) {
		auto src = src_plugin->create_source_from_input(input);
		_max_value += src->get_size();
		_agg.add_input(src);
	}
	/* the number of live TBB tokens in a pipeline does not need to be bigger than the
	 * number of used cores (it was previously set to 5 * cores_number): That multiplier
	 * does not have any impact on the import time but it increases the memory
	 * consumption. On proto-03 (dual hyperthreaded 6-cores with 64 Gio RAM), the
	 * proxy_sample.log file (10 Me) shows that:
	 * - with 5, at most 11.7 Gio are used;
	 * - with 2, at most 7.2 Gio are used;
	 * - with 1, at most 5.3 Gio are used.
	 *
	 * With a mean import time of 240 seconds.
	 *
	 * An other example: a file with 2 columns of 0 makes swap proto-03 at 65 Me (63
	 * Gio used).
	 */

	QSettings& pvconfig = PVCore::PVConfig::get().config();

	int nchunks = pvconfig.value("pvkernel/number_living_chunks", 0).toInt();
	if (nchunks != 0) {
		_chunks = nchunks;
	}
}

PVRush::PVControllerJob_p PVRush::PVExtractor::process_from_agg_nlines(chunk_index start)
{
	chunk_index nlines = _format.get_line_count();
	if (nlines) {
		// Keep user choice.
	} else if (_format.have_grep_filter()) {
		nlines = IMPORT_PIPELINE_ROW_COUNT_LIMIT - start;
	} else {
		// If the NRaw is not compacted, keep uint32_t as a max limite for the number of lines
		nlines = EXTRACTED_ROW_COUNT_LIMIT;
	}

	return process_from_agg_idxes(start, start + nlines - 1);
}

PVRush::PVControllerJob_p PVRush::PVExtractor::process_from_agg_idxes(chunk_index start,
                                                                      chunk_index end)
{
	set_sources_number_fields();
	if (_format.get_line_count() != 0) {
		end = std::min<chunk_index>(end, _format.get_line_count());
	}
	_output.prepare_load(_format);
	_agg.set_skip_lines_count(_format.get_first_line());

	// PVControllerJob_p is a boost shared pointer, that will automatically take care of the
	// deletion of this
	// object when it is not needed anymore !
	PVControllerJob_p job = std::make_shared<PVControllerJob>(
	    start, end, _agg, _chk_flt, _output, _chunks, _format.have_grep_filter());
	job->run_job();

	return job;
}

void PVRush::PVExtractor::set_sources_number_fields()
{
	_agg.set_sources_number_fields(PVCol(_format.get_axes().size()));
}
