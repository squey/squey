/**
 * \file PVSelectionSquareFullParallelView.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVSELECTIONSQUAREFULLPARALLELVIEW_H__
#define __PVSELECTIONSQUAREFULLPARALLELVIEW_H__

#include <pvparallelview/PVSelectionSquare.h>

namespace PVParallelView
{

class PVFullParallelScene;

namespace __impl {

struct PVSelectionBarycenter
{
	PVSelectionBarycenter()
	{
		clear();
	}

	PVZoneID zone_id1;
	PVZoneID zone_id2;
	double factor1;
	double factor2;

	void clear()
	{
		zone_id1 = PVZONEID_INVALID;
		zone_id2 = PVZONEID_INVALID;
		factor1 = 0.0;
		factor2 = 0.0;
	}
};

}

class PVSelectionSquareFullParallelView : public PVSelectionSquare
{
public:
	PVSelectionSquareFullParallelView(Picviz::PVView& view, QGraphicsScene* s);

public:
	void update_position();

public:
	void clear() override;

protected:
	void commit(bool use_selection_modifiers) override;

private:
	void store();

	PVFullParallelScene* scene_parent();
	PVFullParallelScene const* scene_parent() const;

	PVLinesView& get_lines_view();
	PVLinesView const& get_lines_view() const;

	Picviz::PVView& lib_view();
	Picviz::PVView const& lib_view() const;

private:
	__impl::PVSelectionBarycenter _selection_barycenter;
};

}

#endif // __PVSELECTIONSQUAREFULLPARALLELVIEW_H__
