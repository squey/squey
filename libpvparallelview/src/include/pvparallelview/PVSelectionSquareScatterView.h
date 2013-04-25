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

class PVScatterView;

class PVSelectionSquareScatterView : public PVSelectionSquare
{
public:
	PVSelectionSquareScatterView(const PVZoneTree &zt, PVScatterView* sv);

protected:
	void commit(bool use_selection_modifiers) override;
	Picviz::PVView& lib_view() override;

private:
	const PVZoneTree &_zt;
	PVScatterView* _sv;
};

}

#endif // __PVSELECTIONSQUARESCATTERVIEW_H__
