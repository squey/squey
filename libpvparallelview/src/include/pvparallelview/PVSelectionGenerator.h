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

namespace Picviz
{
	class PVSelection;
}

namespace PVParallelView
{
class PVZonesManager;

struct PVLineEqInt
{
	int a;
	int b;
	int c;
	inline int operator()(int X, int Y) const { return a*X+b*Y+c; }
};

struct PVSelectionGenerator
{
	static uint32_t compute_selection_from_rect(PVLinesView& lines_view, PVZoneID zone_id, QRect rect, Picviz::PVSelection& sel);
	static uint32_t compute_selection_from_sliders(PVLinesView& lines_view, PVZoneID zone_id, const typename PVAxisGraphicsItem::selection_ranges_t& ranges, Picviz::PVSelection& sel);
};

}

#endif /* PVSELECTIONGENERATOR_H_ */
