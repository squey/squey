/**
 * \file PVSelectionSquareScatterView.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVSELECTIONSQUARESCATTERVIEW_H__
#define __PVSELECTIONSQUARESCATTERVIEW_H__

#include <pvparallelview/PVSelectionSquare.h>
#include <pvparallelview/PVZoneTree.h>

namespace PVParallelView
{

class PVSelectionSquareScatterView : public PVSelectionSquare
{
public:
	PVSelectionSquareScatterView(Picviz::PVView& view, const PVZoneTree &zt, QGraphicsScene* s) : PVSelectionSquare(view, s), _zt(zt) {};

protected:
	void commit(bool use_selection_modifiers) override
	{
		PVSelectionGenerator::compute_selection_from_scatter_view_rect(_zt, _selection_graphics_item->rect(), _view.get_volatile_selection());
		PVSelectionGenerator::process_selection(_view.shared_from_this(), use_selection_modifiers);
	}

private:
	const PVZoneTree &_zt;
};

}

#endif // __PVSELECTIONSQUARESCATTERVIEW_H__
