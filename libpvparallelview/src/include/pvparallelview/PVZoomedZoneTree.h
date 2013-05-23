/**
 * \file PVZoomedZoneTree.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PARALLELVIEW_PVZOOMEDZONETREE_H
#define PARALLELVIEW_PVZOOMEDZONETREE_H

#include <pvbase/types.h>
#include <pvparallelview/common.h>
#include <picviz/PVPlotted.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVQuadTree.h>
#include <pvparallelview/PVZoneProcessing.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVTLRBuffer.h>

#include <boost/shared_ptr.hpp>

#include <functional>

#include <tbb/tick_count.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#define BROWSE_TBB

// forward declaration
namespace PVCore {
class PVHSVColor;
}

namespace PVParallelView {

// forward declaration
template <size_t Bbits>
class PVBCICode;

/**
 * @class PVZoomedZoneTree
 *
 * This class is the high-level data structure used to store events relatively to a zone.
*]
 * It is firstly designed to be used to zoom in parallel coordinates representations of data sets.
 * It reuse the same structuring than PVZoneTree: a fixed sized array of data structures which
 * contains the events. this array is a first partition of the events space. The lesser level
 * data structure, a quadtree, store and index each partition.
 *
 * The similary between PVZoneTree and PVZoomedZoneTree can be helpful to initialize a PVZoomedZoneTree
 * from its corresponding PVZoneTree: each quadtree can be created from its corresponding PVZoneTree
 *  sub-structure.
 */
class PVZoomedZoneTree
{
	constexpr static size_t bbits = PARALLELVIEW_ZZT_BBITS;
	constexpr static uint32_t mask_int_ycoord = (((uint32_t)1)<<bbits)-1;
	constexpr static size_t quadtree_max_level = 17;

	typedef PVTLRBuffer<bbits> pv_tlr_buffer_t;
	typedef pv_tlr_buffer_t::index_t pv_tlr_index_t;
	typedef PVQuadTree<8192, 1000, 0, bbits> pvquadtree;
	typedef pvquadtree::insert_entry_f insert_entry_f;
	typedef pvquadtree::insert_entry_y1_y2_f insert_entry_y1_y2_f;

	typedef std::function<void(const pvquadtree &tree,
	                           const uint32_t count,
	                           pv_quadtree_buffer_entry_t *buffer,
	                           pv_tlr_buffer_t &tlr,
	                           const insert_entry_f &insert_f)> extract_entries_f;

	typedef std::function<void(
		const pvquadtree &tree,
		PVCore::PVHSVColor* colors,
	    const insert_entry_y1_y2_f &insert_f)>
	extract_entries_y1_y2_f;
	
public:
	typedef constants<bbits> zzt_constants;
	typedef PVBCICode<bbits> pv_bci_code_t;

public:
	/**
	 * @class zzt_tls
	 *
	 * This class is the Thread Local Storage data structure needed by parallel
	 * algorithms used on PVZoomedZoneTree.
	 */
	class zzt_tls
	{
	public:
		/**
		 * Constructor
		 */
		zzt_tls()
		{}

		/**
		 * Destructor
		 */
		~zzt_tls()
		{}

		/**
		 * Getter for the local TLR buffer.
		 */
		pv_tlr_buffer_t &get_tlr_buffer()
		{
			return _tlr_buffer;
		}

		/**
		 * Getter for the local quadtree's bitfield.
		 */
		pv_quadtree_buffer_entry_t *get_quadtree_buffer()
		{
			return &_quadtree_buffer[0];
		}

	private:
		pv_quadtree_buffer_entry_t _quadtree_buffer[QUADTREE_BUFFER_SIZE];
		pv_tlr_buffer_t            _tlr_buffer;
	};

	/**
	 * @class context_t
	 *
	 * This class store the Thread Local Storage data structures needed by parallel
	 * algorithms used on PVZoomedZoneTree.
	 */
	class context_t {
	public:
		typedef tbb::enumerable_thread_specific<zzt_tls> tls_set_t;

		context_t()
		{}

		~context_t()
		{}

		tls_set_t &get_tls() const
		{
			return _tls;
		}
	private:
		mutable tls_set_t _tls;
	};

public:
	/**
	 * Constructor
	 *
	 * @param sel_elts the buffer where PVZoneTree store selected events
	 * @param max_level the depth limit for quadtree
	 */
	PVZoomedZoneTree(const PVRow *sel_elts, const PVRow *bg_elts, uint32_t max_level = quadtree_max_level);

	/**
	 * Destructor
	 */
	~PVZoomedZoneTree();

	/**
	 * Compute the memory used.
	 *
	 * @return the used memory
	 */
	inline size_t memory() const
	{
		size_t mem = sizeof(PVZoomedZoneTree);
		for (uint32_t i = 0; i < NBUCKETS; ++i) {
			mem += _trees[i].memory();
		}
		return mem;
	}

	/**
	 * Tell if this PVZoomedTree has been initialized.
	 *
	 * @return true if it is initialized; false otherwise
	 */
	bool is_initialized() const
	{
		return _initialized;
	}

	/**
	 * Free all internal data structures.
	 */
	void reset();

	/**
	 * Process \zp and \zt to construct the internal data structures.
	 *
	 * @param zp the underlying PVZoneProcessing
	 * @param zt the twin PVZoneTree
	 */
	void process(const PVZoneProcessing &zp, PVZoneTree &zt);

public:
	/**
	 * Sequential processing of \zp to construct the internal data structures.
	 *
	 * @param zp the underlying PVZoneProcessing
	 */
	void process_seq(const PVZoneProcessing &zp);

	/**
	 * Sequential processing of \zt to construct the internal data structures.
	 *
	 * @param zp the underlying PVZoneProcessing
	 * @param zt the twin PVZoneTree
	 */
	__attribute__((noinline)) void process_seq_from_zt(const PVZoneProcessing &zp, PVZoneTree &zt);

	/**
	 * Parallel processing of \zp to construct the internal data structures.
	 *
	 * @param zp the underlying PVZoneProcessing
	 */
	void process_omp(const PVZoneProcessing &zp);

	/**
	 * Parallel processing of \zt to construct the internal data structures.
	 *
	 * @param zp the underlying PVZoneProcessing
	 * @param zt the twin PVZoneTree
	 */
	void process_omp_from_zt(const PVZoneProcessing &zp, PVZoneTree &zt);

	/**
	 * Equality test.
	 *
	 * @param qt the second zoomed zone tree
	 *
	 * @return true if the 2 zoomed zone trees have the same structure and the
	 * same content; false otherwise.
	 */
	bool operator==(PVZoomedZoneTree &zzt) const
	{
		for (size_t i = 0; i < NBUCKETS; ++i) {
			if (_trees[i] == zzt._trees[i]) {
				continue;
			}
			return false;
		}

		return true;
	}

	/**
	 * Save the zoomed zone tree into a file.
	 *
	 * @param filename the output filename
	 *
	 * @return true on success; false otherwise and an error is printed.
	 */
	bool dump_to_file(const char *filename) const;

	/**
	 * Create and load a zoomed zone tree from a file.
	 *
	 * @param filename the input filename
	 *
	 * @return a zoomed zone tree on success; nullptr otherwise and an error is printed.
	 */
	static PVZoomedZoneTree *load_from_file(const char *filename);

	/**
	 * Search for all events whose primary coordinates are in in the range [y1_min,y1_max) and
	 * mark them as selected in \selection.
	 *
	 * @param y1_min the incluse minimal bound along the primary coordinate
	 * @param y1_max the excluse maximal bound along the primary coordinate
	 * @param selection the structure containing the result
	 *
	 * @return the count of selected events in selection
	 */
	inline size_t compute_selection_y1(PVRow t1, const uint64_t y_min, const uint64_t y_max,
	                                   Picviz::PVSelection &selection) const
	{
		if (_initialized == false) {
			return 0;
		}

		size_t num = 0;
		for (uint32_t t2 = 0; t2 < 1024; ++t2) {
			PVRow tree_idx = (t2 * 1024) + t1;
			// if (_sel_elts[tree_idx] != PVROW_INVALID_VALUE) {
				num += _trees[tree_idx].compute_selection_y1(y_min, y_max, selection);
			// }
		}

		return num;
	}

	/**
	 * Search for all events whose secondary coordinates are in the range [y2_min,y2_max) and
	 * mark them as selected in \selection.
	 *
	 * @param y1_min the incluse minimal bound along the primary coordinate
	 * @param y1_max the excluse maximal bound along the primary coordinate
	 * @param selection the structure containing the result
	 *
	 * @return the count of selected events in selection
	 */
	inline size_t compute_selection_y2(PVRow t2, const uint64_t y_min, const uint64_t y_max,
	                                   Picviz::PVSelection &selection) const
	{
		if (_initialized == false) {
			return 0;
		}

		size_t num = 0;
		for (uint32_t t1 = 0; t1 < 1024; ++t1) {
			PVRow tree_idx = (t2 * 1024) + t1;
			// if (_sel_elts[tree_idx] != PVROW_INVALID_VALUE) {
				num += _trees[tree_idx].compute_selection_y2(y_min, y_max, selection);
			// }
		}

		return num;
	}

	/**
	 * Extract all events needed for background image with constraints on primary coordinate.
	 *
	 *
	 * @param ctx the extraction context
	 * @param y_min the incluse minimal bound along the primary coordinate
	 * @param y_max the excluse maximal bound along the primary coordinate
	 * @param y_lim the excluse maximal bound corresponding to the bottom of the resulting image
	 * @param zoom the zoom level
	 * @param width the resulting image's width
	 * @param colors the current color map
	 * @param codes the buffer to store resulting BCI codes
	 * @param beta the horizontal scale factor
	 *
	 * @return the count of found events
	 */
	inline size_t browse_bci_by_y1(context_t &ctx,
	                               uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                               int zoom, uint32_t width,
	                               const PVCore::PVHSVColor* colors,
	                               pv_bci_code_t* codes,
	                               const float beta = 1.0f) const
	{
		if (_initialized == false) {
			return 0;
		}

#ifdef BROWSE_TBB
		return browse_trees_bci_by_y1_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y1_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y2_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      pv_tlr_buffer_t &tlr,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_from_y1(y_min, y_max,
			                                                         zoom, y2_count,
			                                                         buffer, insert_f, tlr);
		                                  },
		                                  colors, codes, beta);
	}



	inline void browse_bci_by_y1_y2(
		uint64_t y1_min,
		uint64_t y1_max,
		uint64_t y2_min,
		uint64_t y2_max,
		int zoom,
		double alpha,
		PVCore::PVHSVColor const* const colors,
		PVCore::PVHSVColor* const image,
		uint32_t image_width,
		tbb::task_group_context* tbb_ctxt = nullptr
	) const
	{
		browse_trees_bci_by_y1_y2_tbb(y1_min, y1_max, y2_min, y2_max, zoom, alpha, colors, image, image_width,
			// extract_entries_y1_y2_f:
			[&](const pvquadtree &tree,
				PVCore::PVHSVColor* image,
			  const insert_entry_y1_y2_f &insert_f)
			{
				tree.get_first_from_y1_y2(y1_min, y1_max, y2_min, y2_max, zoom, alpha, image, insert_f);
			}, nullptr, tbb_ctxt);
	}

	inline void browse_bci_by_y1_y2_sel(
		uint64_t y1_min,
		uint64_t y1_max,
		uint64_t y2_min,
		uint64_t y2_max,
		int zoom,
		double alpha,
		PVCore::PVHSVColor const* const colors,
		PVCore::PVHSVColor* const image,
		Picviz::PVSelection const& sel,
		uint32_t image_width,
		tbb::task_group_context* tbb_ctxt = nullptr
	) const
	{
		browse_trees_bci_by_y1_y2_tbb(y1_min, y1_max, y2_min, y2_max, zoom, alpha, colors, image, image_width,
			// extract_entries_y1_y2_f:
			[&](const pvquadtree &tree,
				PVCore::PVHSVColor* image,
			  const insert_entry_y1_y2_f &insert_f)
			{
				tree.get_first_from_y1_y2_sel(y1_min, y1_max, y2_min, y2_max, zoom, alpha, image, insert_f, sel);
			}, _sel_elts, tbb_ctxt);
	}


	/**
	 * Extract all events needed for background image with constraints on secondary coordinate.
	 *
	 *
	 * @param ctx the extraction context
	 * @param y_min the incluse minimal bound along the secondary coordinate
	 * @param y_max the excluse maximal bound along the secondary coordinate
	 * @param y_lim the excluse maximal bound corresponding to the bottom of the resulting image
	 * @param zoom the zoom level
	 * @param width the resulting image's width
	 * @param colors the current color map
	 * @param codes the buffer to store resulting BCI codes
	 * @param beta the horizontal scale factor
	 *
	 * @return the count of found events
	 */
	inline size_t browse_bci_by_y2(context_t &ctx,
	                               uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                               int zoom, uint32_t width,
	                               const PVCore::PVHSVColor *colors,
	                               pv_bci_code_t *codes,
	                               const float beta = 1.0f) const
	{
		if (_initialized == false) {
			return 0;
		}

#ifdef BROWSE_TBB
		return browse_trees_bci_by_y2_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y2_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y1_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      pv_tlr_buffer_t &tlr,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_from_y2(y_min, y_max,
			                                                         zoom, y1_count,
			                                                         buffer, insert_f, tlr);
		                                  },
		                                  colors, codes, beta);
	}

	/**
	 * Extract all events needed for selection image with constraints on primary coordinate.
	 *
	 *
	 * @param ctx the extraction context
	 * @param y_min the incluse minimal bound along the primary coordinate
	 * @param y_max the excluse maximal bound along the primary coordinate
	 * @param y_lim the excluse maximal bound corresponding to the bottom of the resulting image
	 * @param zoom the zoom level
	 * @param width the resulting image's width
	 * @param colors the current color map
	 * @param codes the buffer to store resulting BCI codes
	 * @param beta the horizontal scale factor
	 *
	 * @return the count of found events
	 */
	inline size_t browse_bci_sel_by_y1(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   const Picviz::PVSelection &selection,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
		if (_initialized == false) {
			return 0;
		}

#ifdef BROWSE_TBB
		return browse_trees_bci_by_y1_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y1_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y2_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      pv_tlr_buffer_t &tlr,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_sel_from_y1(y_min, y_max,
			                                                             selection,
			                                                             zoom, y2_count,
			                                                             buffer, insert_f, tlr);
		                                  },
		                                  colors, codes, beta, _sel_elts);
	}

	/**
	 * Extract all events needed for selection image with constraints on secondary coordinate.
	 *
	 *
	 * @param ctx the extraction context
	 * @param y_min the incluse minimal bound along the secondary coordinate
	 * @param y_max the excluse maximal bound along the secondary coordinate
	 * @param y_lim the excluse maximal bound corresponding to the bottom of the resulting image
	 * @param zoom the zoom level
	 * @param width the resulting image's width
	 * @param colors the current color map
	 * @param codes the buffer to store resulting BCI codes
	 * @param beta the horizontal scale factor
	 *
	 * @return the count of found events
	 */
	inline size_t browse_bci_sel_by_y2(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   const Picviz::PVSelection &selection,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
		if (_initialized == false) {
			return 0;
		}

#ifdef BROWSE_TBB
		return browse_trees_bci_by_y2_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y2_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y1_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      pv_tlr_buffer_t &tlr,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_sel_from_y2(y_min, y_max,
			                                                             selection,
			                                                             zoom, y1_count,
			                                                             buffer, insert_f, tlr);
		                                  },
		                                  colors, codes, beta, _sel_elts);
	}

	inline size_t browse_bci_bg_by_y1(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   const Picviz::PVSelection &unselected,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
		if (_initialized == false) {
			return 0;
		}

#ifdef BROWSE_TBB
		return browse_trees_bci_by_y1_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y1_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y2_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      pv_tlr_buffer_t &tlr,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  size_t ret = tree.get_first_sel_from_y1(y_min, y_max,
			                                                             unselected,
			                                                             zoom, y2_count,
			                                                             buffer, insert_f, tlr);
											  if (ret == 0) {
												  tree.get_first_from_y1(y_min, y_max,
																		 zoom, y2_count,
																		 buffer, insert_f, tlr);
											  }
		                                  },
		                                  colors, codes, beta, _bg_elts);
	}

	inline size_t browse_bci_bg_by_y2(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   const Picviz::PVSelection &unselected,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
		if (_initialized == false) {
			return 0;
		}

#ifdef BROWSE_TBB
		return browse_trees_bci_by_y2_tbb(ctx, y_min, y_max, y_lim, zoom, width,
#else
		return browse_trees_bci_by_y2_seq(ctx, y_min, y_max, y_lim, zoom, width,
#endif
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y1_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      pv_tlr_buffer_t &tlr,
		                                      const insert_entry_f &insert_f)
		                                  {
										      size_t ret = tree.get_first_sel_from_y2(y_min, y_max,
			                                                             unselected,
			                                                             zoom, y1_count,
			                                                             buffer, insert_f, tlr);
											  if (ret == 0) {
												  tree.get_first_from_y2(y_min, y_max,
																		 zoom, y1_count,
																		 buffer, insert_f, tlr);
											  }
			                            
		                                  },
		                                  colors, codes, beta, _bg_elts);
	}


	/**
	 * Test function for sequential implementation of browse_bci_by_y1
	 *
	 */
	inline size_t browse_bci_by_y1_seq(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
		if (_initialized == false) {
			PVLOG_WARN("you forget to initialize the ZoomedZoneTree\n");
			return 0;
		}

		return browse_trees_bci_by_y1_seq(ctx, y_min, y_max, y_lim, zoom, width,
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y2_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      pv_tlr_buffer_t &tlr,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_from_y1(y_min, y_max,
			                                                         zoom, y2_count,
			                                                         buffer, insert_f, tlr);
		                                  },
		                                  colors, codes, beta);
	}

	/**
	 * Test function for parallel implementation of browse_bci_by_y1
	 *
	 */
	inline size_t browse_bci_by_y1_tbb(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
		if (_initialized == false) {
			PVLOG_WARN("you forget to initialize the ZoomedZoneTree\n");
			return 0;
		}

		return browse_trees_bci_by_y1_tbb(ctx, y_min, y_max, y_lim, zoom, width,
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y2_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      pv_tlr_buffer_t &tlr,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_from_y1(y_min, y_max,
			                                                         zoom, y2_count,
			                                                         buffer, insert_f, tlr);
		                                  },
		                                  colors, codes, beta);
	}

	/**
	 * Test function for sequential implementation of browse_bci_by_y2
	 *
	 */
	inline size_t browse_bci_by_y2_seq(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
		if (_initialized == false) {
			PVLOG_WARN("you forget to initialize the ZoomedZoneTree\n");
			return 0;
		}

		return browse_trees_bci_by_y2_seq(ctx, y_min, y_max, y_lim, zoom, width,
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y1_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      pv_tlr_buffer_t &tlr,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_from_y2(y_min, y_max,
			                                                         zoom, y1_count,
			                                                         buffer, insert_f, tlr);
		                                  },
		                                  colors, codes, beta);
	}

	/**
	 * Test function for parallel implementation of browse_bci_by_y2
	 *
	 */
	inline size_t browse_bci_by_y2_tbb(context_t &ctx,
	                                   uint64_t y_min, uint64_t y_max, uint64_t y_lim,
	                                   int zoom, uint32_t width,
	                                   const PVCore::PVHSVColor *colors,
	                                   pv_bci_code_t *codes,
	                                   const float beta = 1.0f) const
	{
		if (_initialized == false) {
			PVLOG_WARN("you forget to initialize the ZoomedZoneTree\n");
			return 0;
		}

		return browse_trees_bci_by_y2_tbb(ctx, y_min, y_max, y_lim, zoom, width,
		                                  [&](const pvquadtree &tree,
		                                      const uint32_t y1_count,
		                                      pv_quadtree_buffer_entry_t *buffer,
		                                      pv_tlr_buffer_t &tlr,
		                                      const insert_entry_f &insert_f)
		                                  {
			                                  tree.get_first_from_y2(y_min, y_max,
			                                                         zoom, y1_count,
			                                                         buffer, insert_f, tlr);
		                                  },
		                                  colors, codes, beta);
	}

	void compute_min_indexes_sel(Picviz::PVSelection const& sel);

private:
	/**
	 * Sequential implementation used by browse_bci_by_y2 and browse_bci_sel_by_y2.
	 *
	 */
	size_t browse_trees_bci_by_y1_seq(context_t &ctx,
	                                  uint64_t y_min, uint64_t y_max, uint64_t y_lim, int zoom,
	                                  uint32_t width,
	                                  const extract_entries_f &extract_f,
	                                  const PVCore::PVHSVColor *colors, pv_bci_code_t *codes,
	                                  const float beta = 1.0f,
	                                  const PVRow* sel_elts = nullptr) const;

	/**
	 * Sequential implementation used by browse_bci_by_y2 and browse_bci_sel_by_y2.
	 *
	 */
	size_t browse_trees_bci_by_y2_seq(context_t &ctx,
	                                  uint64_t y_min, uint64_t y_max, uint64_t y_lim, int zoom,
	                                  uint32_t width,
	                                  const extract_entries_f &extract_f,
	                                  const PVCore::PVHSVColor *colors, pv_bci_code_t *codes,
	                                  const float beta = 1.0f,
	                                  const PVRow* sel_elts = nullptr) const;

	/**
	 * Parallel implementation used by browse_bci_by_y1 and browse_bci_sel_by_y1.
	 *
	 */
	size_t browse_trees_bci_by_y1_tbb(context_t &ctx,
	                                  uint64_t y_min, uint64_t y_max, uint64_t y_lim, int zoom,
	                                  uint32_t width,
	                                  const extract_entries_f &extract_f,
	                                  const PVCore::PVHSVColor *colors, pv_bci_code_t *codes,
	                                  const float beta = 1.0f,
	                                  const PVRow* sel_elts = nullptr) const;

	/**
	 * Parallel implementation used by browse_bci_by_y2 and browse_bci_sel_by_y2.
	 *
	 */
	size_t browse_trees_bci_by_y2_tbb(context_t &ctx,
	                                  uint64_t y_min, uint64_t y_max, uint64_t y_lim, int zoom,
	                                  uint32_t width,
	                                  const extract_entries_f &extract_f,
	                                  const PVCore::PVHSVColor *colors, pv_bci_code_t *codes,
	                                  const float beta = 1.0f,
	                                  const PVRow* sel_elts = nullptr) const;

	void browse_trees_bci_by_y1_y2_tbb(
		uint64_t y1_min,
		uint64_t y1_max,
		uint64_t y2_min,
		uint64_t y2_max,
		int zoom,
		double alpha,
		PVCore::PVHSVColor const* const colors,
		PVCore::PVHSVColor* const image,
		uint32_t image_width,
		const extract_entries_y1_y2_f &extract_f,
		PVRow const* const sel_elts = nullptr,
		tbb::task_group_context* tbb_ctxt = nullptr
	) const;

	/**
	 * Compute the index in the quadtee forest given 2 coordinates.
	 *
	 * @param y1 the primary coordinate
	 * @param y1 the secondary coordinate
	 */
	inline uint32_t compute_index(uint32_t y1, uint32_t y2) const
	{
		return  (((y2 >> (32-NBITS_INDEX)) & MASK_INT_YCOORD) << NBITS_INDEX) +
			((y1 >> (32-NBITS_INDEX)) & MASK_INT_YCOORD);
	}

	/**
	 * Compute the index in the quadtee forest given one quadtree's internal representation of event.
	 *
	 * @param e the quadtree's internal representation of event
	 */
	inline uint32_t compute_index(const PVParallelView::PVQuadTreeEntry &e) const
	{
		return compute_index(e.y1, e.y2);
	}

	void init_structures();

private:
	pvquadtree      *_trees;
	const PVRow     *_sel_elts;
	const PVRow     *_bg_elts;
	uint32_t         _max_level;
	bool             _initialized;
};

typedef boost::shared_ptr<PVZoomedZoneTree> PVZoomedZoneTree_p;

}

#endif //  PARALLELVIEW_PVZOOMEDZONETREE_H
