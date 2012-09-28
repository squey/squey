/**
 * \file PVExtractor.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVRUSH_PVEXTRACTOR_FILE_H
#define PVRUSH_PVEXTRACTOR_FILE_H

#include <pvbase/general.h>

#include <pvbase/types.h>
#include <pvkernel/rush/PVAggregator.h>
#include <pvkernel/rush/PVController.h>
#include <pvkernel/rush/PVControllerThread.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVPureMappingProcessing.h>
#include <pvkernel/rush/PVRawSourceBase_types.h>

namespace PVRush {

// The famous and wanted PVExtractor !!!!
/*! \brief Extract datas from an aggregator, process them through filters and write the result to an NRaw
 *
 * This class owns an aggregator and a NRaw (see PVRush::PVNraw). Given a chunk filter, it process a given number
 * of lines and write them to its internal NRaw.
 *
 * It also owns a PVRush::PVController object, that is used to launch and/or cancel running jobs. A priority system allows a job
 * to be the next one running. In order to work, start_controller need to be called.
 *
 * \note We could also imagine that a global PVController object would be used for all PVExtractor's, but that's not our choice for now.
 */
class LibKernelDecl PVExtractor {
public:
	PVExtractor(unsigned int nchunks = 0);
	~PVExtractor();
public:
	/*! \brief Launch the internal job controller
	 */
	void start_controller();

	/*! \brief Gracefully stop the internal job controller
	 * This will stop the controller after the end of current job.
	 */
	void gracefully_stop_controller();
	
	/*! \brief Stop the internal job controller
	 * This will cancel the current job and stop the controller.
	 */
	void force_stop_controller();

	/*! \brief Add a PVRawSourceBase to the internal aggregator
	 * This function adds a source to the internal aggregator.
	 */
	void add_source(PVRush::PVRawSourceBase_p src);

	/*! \brief Set the chunk filter used during the extraction
	 * \param[in] chk_flt A boost::function object of the corresponding PVChunkFilter (can be easily obtained via the PVChunkFilter::f() method)
	 *
	 * \note It is the responsability of the caller to have the pointer to the original PVChunkFilter object valid.
	 */
	void set_chunk_filter(PVFilter::PVChunkFilter_f chk_flt);

	/*! \brief 
	 *  \return A PVControllerJob object that represent the job that has been pushed to the internal job controller. It can be used by the caller to wait for the end of the job (see PVControllerJob::wait_end).
	 */
	PVControllerJob_p process_from_pvrow(PVRow start, PVRow end, int priority = 0, bool force_process = true);

	/*! \brief Process a given number of lines from a given index
	 *  \param[in] start Index to start the extraction from (an index is typically a line number).
	 *  \param[in] nlines Number of lines to extract. It is 
	 *  \param[in] priority Priority of the job
	 *  \return A PVControllerJob object that represent the job that has been pushed to the internal job controller. It can be used by the caller to wait for the end of the job (see PVControllerJob::wait_end).
	 *  \sa PVAggregator
	 */
	PVControllerJob_p process_from_agg_nlines(chunk_index start, chunk_index nlines, int priority = 0);

	PVControllerJob_p process_from_agg_nlines_last_param() { return process_from_agg_nlines(_last_start, _last_nlines); }

	/*! \brief Process param[in]s between indexes "start" and "end"
	 *  \param[in] start Index to start the extraction from (an index is typically a line number).
	 *  \param[in] end Index to end the extraction at
	 *  \param[in] priority Priority of the job
	 *  \return A PVControllerJob object that represent the job that has been pushed to the internal job controller. It can be used by the caller to wait for the end of the job (see PVControllerJob::wait_end).
	 *  \sa PVAggregator
	 */
	PVControllerJob_p process_from_agg_idxes(chunk_index start, chunk_index end, int priority = 0);

	/*! \brief 
	 *  \param[in] priority Priority of the job
	 *  \return A PVControllerJob object that represent the job that has been pushed to the internal job controller. It can be used by the caller to wait for the end of the job (see PVControllerJob::wait_end).
	 *  \sa PVAggregator
	 */
	PVControllerJob_p read_everything(int priority = 0);

	/*! \brief Get the list of sources of the internal aggregator
	 */
	PVAggregator::list_inputs const& get_inputs() const;

	/*! \brief Get a reference to the internal aggregator
	 */
	PVAggregator& get_agg();

	/*! \brief Get a reference to the internal NRaw
	 */
	PVNraw& get_nraw();

	/*! \brief Get a reference to the internal PVFormat of the internal NRaw
	 */
	PVFormat& get_format();
	PVFormat const& get_format() const;

	/*! \brief Set the format of the NRaw
	 */
	void set_format(PVFormat const& format);

	/*! \brief Save a copy of the current NRaw
	 *
	 * Save a copy of the current NRaw. If a copy has already been saved, it is ereased by this one.
	 *  \sa restore_nraw
	 *  \sa clear_save_nraw
	 */
	void save_nraw();

	/*! \brief Restore a copy of the NRaw previously saved by save_nraw
	 *
	 * Restore a copy of the NRaw previously saved thanks to save_nraw. If no NRaw has been saved, this
	 * function does nothing.
	 */
	void restore_nraw();

	/*! \brief Clear the NRaw previously saved by save_nraw
	 *
	 * Clear the NRaw previously saved thanks to save_nraw. If no NRaw has been saved, this function does nothing.
	 */
	void clear_saved_nraw();

	/*! \brief Get the number of axes expected by the internal format.
	 */
	PVCol get_number_axes();

	void force_number_axes(PVCol naxes);

	chunk_index get_last_start() { return _last_start; }
	chunk_index get_last_nlines() { return _last_nlines; }
	void set_last_start(chunk_index start) { _last_start = start; }
	void set_last_nlines(chunk_index nlines) { _last_nlines = nlines; }
	inline void set_number_living_chunks(unsigned int nchunks)
	{
		if (nchunks > 0) {
			_chunks = nchunks;
		}
	}

	void dump_inv_elts(bool dump) { _dump_inv_elts = dump; }
	void dump_all_elts(bool dump) { _dump_all_elts = dump; }

	static PVCore::PVArgumentList default_args_extractor();

	void dump_mapnraw();
	void dump_nraw();
	void debug();

public:
	inline PVNrawOutput::list_chunk_functions& chunk_functions() { return _out_nraw.chunk_functions(); }
	inline PVNrawOutput::list_chunk_functions const& chunk_functions() const { return _out_nraw.chunk_functions(); }

	inline PVFilter::PVPureMappingProcessing::list_pure_mapping_t& pure_mapping_functions() { return _mapping_flt.pure_mappings(); } 
	inline PVFilter::PVPureMappingProcessing::list_pure_mapping_t const& pure_mapping_functions() const { return _mapping_flt.pure_mappings(); } 

private:
	void set_sources_number_fields();
	
protected:
	/*! \brief Find the aggregator index of a line present in the internal nraw
	 */
	chunk_index pvrow_to_agg_index(PVRow start, bool& found);

protected:
	PVAggregator _agg;
	PVNraw _nraw;
	PVNraw _saved_nraw;
	bool _saved_nraw_valid;
	PVController _ctrl;
	PVControllerThread _ctrl_th;
	PVNrawOutput _out_nraw; // Linked to _nraw
	PVFilter::PVPureMappingProcessing _mapping_flt;
	PVFilter::PVChunkFilter_f _chk_flt;
	unsigned int _chunks;
	bool _dump_inv_elts;
	bool _dump_all_elts;
	PVCol _force_naxes; 

protected:
	chunk_index _last_start;
	chunk_index _last_nlines;
	
};

}

#endif
