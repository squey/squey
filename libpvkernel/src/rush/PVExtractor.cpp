/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <pvkernel/core/debug.h>
#include <iostream>

#include <tbb/task_scheduler_init.h>

PVRush::PVExtractor::PVExtractor(unsigned int chunks) :
	_nraw(new PVRush::PVNraw()),
	_out_nraw(*_nraw),
	_chunks(chunks),
	_dump_inv_elts(false),
	_dump_all_elts(false),
	_force_naxes(0),
	_last_start(0),
	_last_nlines(1)
{
	if (chunks == 0) {
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
		_chunks = tbb::task_scheduler_init::default_num_threads();
		PVLOG_DEBUG("(PVExtractor::PVExtractor) using %d chunks\n", _chunks);
	}
}

void PVRush::PVExtractor::add_source(PVRush::PVRawSourceBase_p src)
{
	//TODO: check if controller is running a job
	_agg.add_input(src);
}

void PVRush::PVExtractor::set_chunk_filter(PVFilter::PVChunkFilter_f chk_flt)
{
	_chk_flt = chk_flt;
}

PVRush::PVFormat& PVRush::PVExtractor::get_format()
{
	return *get_nraw().get_format();
}

const PVRush::PVFormat& PVRush::PVExtractor::get_format() const
{
	return *get_nraw().get_format();
}

PVRush::PVControllerJob_p PVRush::PVExtractor::process_from_agg_nlines(chunk_index start, chunk_index nlines)
{
	nlines = std::min(nlines, (chunk_index) INENDI_LINES_MAX);

	set_sources_number_fields();
	get_nraw().prepare_load(nlines);

	_agg.set_skip_lines_count(start);
	_agg.set_strict_mode(start > 0);

	// PVControllerJob_p is a boost shared pointer, that will automatically take care of the deletion of this
	// object when it is not needed anymore !
	PVControllerJob_p job = PVControllerJob_p(new PVControllerJob(start, 0, nlines, PVControllerJob::sc_n_elts,
				_agg, _chk_flt, _out_nraw, _chunks, _dump_inv_elts, _dump_all_elts));
	job->run_job();	

	_last_start = start;
	_last_nlines = nlines;

	return job;
}

PVRush::PVControllerJob_p PVRush::PVExtractor::process_from_agg_idxes(chunk_index start, chunk_index end)
{
	end = std::min(end, start + ((chunk_index) INENDI_LINES_MAX) - 1);

	set_sources_number_fields();
	get_nraw().prepare_load(end-start);

	// PVControllerJob_p is a boost shared pointer, that will automatically take care of the deletion of this
	// object when it is not needed anymore !
	PVControllerJob_p job = PVControllerJob_p(new PVControllerJob(start, end, 0, PVControllerJob::sc_idx_end,
				_agg, _chk_flt, _out_nraw, _chunks, _dump_inv_elts, _dump_all_elts));
	job->run_job();
	
	return job;
}

PVRush::PVControllerJob_p PVRush::PVExtractor::read_everything()
{
	PVControllerJob_p job = PVControllerJob_p(new PVControllerJob(0, 0, 0, PVControllerJob::sc_idx_end,
				_agg, _chk_flt, _out_nraw, _chunks, false, false));
	job->run_read_all_job();

	return job;
}

void PVRush::PVExtractor::dump_nraw()
{
	PVLOG_INFO("Nraw:\n");
	for (size_t i = 0; i < inendi_min(10,get_nraw().get_row_count()); i++) {
		PVLOG_INFO("Line %d: ", i);
		for (int j = 0; j < get_nraw().get_number_cols(); j++) {
			std::cerr << get_nraw().at_string(i,j) << ",";
		}
		std::cerr << std::endl;
	}
}

PVRush::PVAggregator::list_inputs const& PVRush::PVExtractor::get_inputs() const
{
	return _agg.get_inputs();
}

PVRush::PVAggregator& PVRush::PVExtractor::get_agg()
{
	return _agg;
}

void PVRush::PVExtractor::debug()
{
	PVLOG_DEBUG("PVExtractor debug\n");
	_agg.debug();
	PVLOG_DEBUG("PVExtractor nraw\n");
	dump_nraw();
}

void PVRush::PVExtractor::reset_nraw()
{
	PVRush::PVFormat_p format = _nraw->get_format();
	_nraw.reset(new PVNraw());
	_nraw->set_format(format);
	_out_nraw.set_nraw_dest(*_nraw);
}

void PVRush::PVExtractor::set_format(PVFormat const& format)
{
	get_nraw().set_format(std::make_shared<PVFormat>(format));
}

void PVRush::PVExtractor::force_number_axes(PVCol naxes)
{
	_force_naxes = naxes;
}

PVCol PVRush::PVExtractor::get_number_axes()
{
	if (get_nraw().get_format()) {
		return get_nraw().get_format()->get_axes().size();
	}
	
	// The number of axes is unknown, the NRAW will be resized
	// when the first line is created (see PVNraw::add_row)
	return _force_naxes;
}

void PVRush::PVExtractor::set_sources_number_fields()
{
	_agg.set_sources_number_fields(get_number_axes());
}

PVCore::PVArgumentList PVRush::PVExtractor::default_args_extractor()
{
	PVCore::PVArgumentList args;
	args["inv_elts"] = false;
	return args;
}
