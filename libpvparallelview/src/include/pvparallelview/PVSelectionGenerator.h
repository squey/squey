/**
 * \file PVSelectionGenerator.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVSELECTIONGENERATOR_H_
#define PVSELECTIONGENERATOR_H_

#include <pvparallelview/common.h>
#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVLinesView.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVCallHelper.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVObserverSignal.h>

namespace Picviz
{
class PVSelection;
class PVView;
typedef PVCore::PVSharedPtr<PVView> PVView_sp;
}

namespace PVParallelView
{
class PVZonesManager;
class PVZoneTree;
class PVHitGraphBlocksManager;

struct PVLineEqInt
{
	int a;
	int b;
	int c;
	inline int operator()(int X, int Y) const { return a*X+b*Y+c; }
};

struct PVSelectionGenerator
{
	static uint32_t compute_selection_from_parallel_view_rect(PVLinesView& lines_view, PVZoneID zone_id, QRect rect, Picviz::PVSelection& sel);
	static uint32_t compute_selection_from_parallel_view_sliders(PVLinesView& lines_view, PVZoneID zone_id, const typename PVAxisGraphicsItem::selection_ranges_t& ranges, Picviz::PVSelection& sel);


	static uint32_t compute_selection_from_scatter_view_rect(
		const uint32_t* y1_plotted,
		const uint32_t* y2_plotted,
		const PVRow nrows,
		const QRectF& rect,
		Picviz::PVSelection& sel,
		Picviz::PVSelection const& layers_sel
	);
	static uint32_t compute_selection_from_scatter_view_rect_plotted_seq(
		const uint32_t* y1_plotted,
		const uint32_t* y2_plotted,
		const PVRow nrows,
		const QRectF& rect,
		Picviz::PVSelection& sel,
		Picviz::PVSelection const& layers_sel
	);
	static uint32_t compute_selection_from_scatter_view_rect_plotted_sse(
		const uint32_t* y1_plotted,
		const uint32_t* y2_plotted,
		const PVRow nrows,
		const QRectF& rect,
		Picviz::PVSelection& sel,
		Picviz::PVSelection const& layers_sel
	);

	static uint32_t compute_selection_from_hit_count_view_rect(const PVHitGraphBlocksManager& manager,
	                                                           const QRectF& rect,
	                                                           const uint32_t max_count,
	                                                           Picviz::PVSelection& sel)
	{ return compute_selection_from_hit_count_view_rect_sse_invariant_omp(manager, rect, max_count, sel); }

	static void process_selection(Picviz::PVView_sp view, bool use_modifiers = true);

	// Implementations
	static uint32_t compute_selection_from_hit_count_view_rect_serial(const PVHitGraphBlocksManager& manager,
	                                                                  const QRectF& rect,
	                                                                  const uint32_t max_count,
	                                                                  Picviz::PVSelection& sel);
	static uint32_t compute_selection_from_hit_count_view_rect_serial_invariant(const PVHitGraphBlocksManager& manager,
	                                                                  const QRectF& rect,
	                                                                  const uint32_t max_count,
	                                                                  Picviz::PVSelection& sel);
	static uint32_t compute_selection_from_hit_count_view_rect_sse(const PVHitGraphBlocksManager& manager,
	                                                               const QRectF& rect,
	                                                               const uint32_t max_count,
	                                                               Picviz::PVSelection& sel);
	static uint32_t compute_selection_from_hit_count_view_rect_sse_invariant_omp(
			const PVHitGraphBlocksManager& manager,
			const QRectF& rect,
			const uint32_t max_count,
			Picviz::PVSelection& sel);
};

}

#endif /* PVSELECTIONGENERATOR_H_ */
