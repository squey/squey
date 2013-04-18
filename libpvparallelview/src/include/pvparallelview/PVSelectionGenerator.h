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
	static uint32_t compute_selection_from_scatter_view_rect(PVZoneTree const& ztree, QRectF rect, Picviz::PVSelection& sel);

	static void process_selection(Picviz::PVView_sp view, bool use_modifiers = true);
};

}

#endif /* PVSELECTIONGENERATOR_H_ */
