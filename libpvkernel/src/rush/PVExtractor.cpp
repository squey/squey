/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <tbb/task_scheduler_init.h>

PVRush::PVExtractor::PVExtractor()
    : _nraw(new PVRush::PVNraw())
    , _out_nraw(*_nraw)
    , _chunks(tbb::task_scheduler_init::default_num_threads())
    , _force_naxes(0)
    , _last_start(0)
    , _last_nlines(1)
{
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
}

void PVRush::PVExtractor::add_source(PVRush::PVRawSourceBase_p src)
{
	_agg.add_input(src);
}

void PVRush::PVExtractor::set_chunk_filter(PVFilter::PVChunkFilterByElt* chk_flt)
{
	_chk_flt = chk_flt;
}

PVRush::PVFormat& PVRush::PVExtractor::get_format()
{
	return _format;
}

const PVRush::PVFormat& PVRush::PVExtractor::get_format() const
{
	return _format;
}

PVRush::PVControllerJob_p PVRush::PVExtractor::process_from_agg_nlines(chunk_index start)
{
	set_sources_number_fields();

	chunk_index nlines = _format.get_line_count();
	nlines = (nlines) ? nlines : std::numeric_limits<uint32_t>::max();

	get_nraw().prepare_load(_format.get_storage_format());

	_agg.set_skip_lines_count(_format.get_first_line());

	// PVControllerJob_p is a boost shared pointer, that will automatically take care of the
	// deletion of this
	// object when it is not needed anymore !
	PVControllerJob_p job = PVControllerJob_p(
	    new PVControllerJob(start, 0, nlines, PVControllerJob::sc_n_elts, _agg, *_chk_flt,
	                        _out_nraw, _chunks, _format.have_grep_filter()));
	job->run_job();

	_last_start = start;
	_last_nlines = nlines;

	return job;
}

PVRush::PVControllerJob_p PVRush::PVExtractor::process_from_agg_idxes(chunk_index start,
                                                                      chunk_index end)
{
	set_sources_number_fields();
	if (_format.get_line_count() != 0) {
		end = std::min<chunk_index>(end, _format.get_line_count());
	}
	get_nraw().prepare_load(_format.get_storage_format());
	_agg.set_skip_lines_count(_format.get_first_line());

	// PVControllerJob_p is a boost shared pointer, that will automatically take care of the
	// deletion of this
	// object when it is not needed anymore !
	PVControllerJob_p job = PVControllerJob_p(new PVControllerJob(
	    start, end, 0, PVControllerJob::sc_idx_end, _agg, *_chk_flt, _out_nraw, _chunks, false));
	job->run_job();

	return job;
}

void PVRush::PVExtractor::reset_nraw()
{
	_nraw.reset(new PVNraw());
	_out_nraw.set_nraw_dest(*_nraw);
}

void PVRush::PVExtractor::set_format(PVFormat const& format)
{
	_format = format;
}

void PVRush::PVExtractor::force_number_axes(PVCol naxes)
{
	_force_naxes = naxes;
}

void PVRush::PVExtractor::set_sources_number_fields()
{
	_agg.set_sources_number_fields(_format.get_axes().size());
}

PVCore::PVArgumentList PVRush::PVExtractor::default_args_extractor()
{
	PVCore::PVArgumentList args;
	args["inv_elts"] = false;
	return args;
}
