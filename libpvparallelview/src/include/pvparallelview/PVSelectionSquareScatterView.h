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
	PVSelectionSquareScatterView(const uint32_t* y1_plotted, const uint32_t* y2_plotted, const PVRow nrows, PVScatterView* sv);

protected:
	void commit(bool use_selection_modifiers) override;
	Picviz::PVView& lib_view() override;

private:
	const uint32_t* _y1_plotted;
	const uint32_t* _y2_plotted;
	const PVRow _nrows;
	PVScatterView* _sv;
};

}

#endif // __PVSELECTIONSQUARESCATTERVIEW_H__
