/**
 * \file PVSelectionGenerator.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVSELECTIONGENERATOR_H_
#define PVSELECTIONGENERATOR_H_

#include <pvparallelview/common.h>
#include <pvparallelview/PVAxisGraphicsItem.h>

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

class PVSelectionGenerator
{
public:
	PVSelectionGenerator(PVZonesManager& zm) : _zm(zm) {};

	uint32_t compute_selection_from_rect(PVZoneID zid, QRect rect, Picviz::PVSelection& sel);
	uint32_t compute_selection_from_sliders(PVZoneID zid, const typename PVAxisGraphicsItem::selection_ranges_t& ranges, Picviz::PVSelection& sel);

private:
	PVZonesManager& _zm;
};

}

#endif /* PVSELECTIONGENERATOR_H_ */