/**
 * \file PVZoomedZoneView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZOOMEDZONEVIEW_H
#define PVPARALLELVIEW_PVZOOMEDZONEVIEW_H

#include <vector>

#include <pvbase/types.h>
#include <pvkernel/core/PVAlgorithms.h>

namespace PVParallelView {

class PVZonesDrawing;

class PVZoomedZoneView
{
public:
	PVZoomedZoneView(PVZonesDrawing &zones_drawing, PVCol axis, bool zone_after_axis) :
		_zones_drawing(zones_drawing),
		_axis(axis),
		_zone_after_axis(zone_after_axis)
	{
	}

	void translate(int32_t view_y)
	{
		// TODO
		(void) view_y;
	}

	void render_all()
	{
		// TODO
	}

private:
	PVZonesDrawing &_zones_drawing;
	PVCol           _axis;
	PVRow           _top;             // the top position of the viewport
	PVRow           _bottom;          // the bottom position of the viewport
	bool            _zone_after_axis;
};

}

#endif // PVPARALLELVIEW_PVZOOMEDZONEVIEW_H
