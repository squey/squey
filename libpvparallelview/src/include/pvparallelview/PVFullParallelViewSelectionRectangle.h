
/**
 * \file PVFullParallelViewSelectionRectangle.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef PVPARALLELVIEW_PVFULLPARALLELVIEWSELECTIONSQUARE_H
#define PVPARALLELVIEW_PVFULLPARALLELVIEWSELECTIONSQUARE_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVSelectionRectangle.h>

#include <pvbase/types.h>

namespace PVParallelView
{

class PVFullParallelScene;
class PVLinesView;

class PVFullParallelViewSelectionRectangle : public PVSelectionRectangle
{
public:
	struct barycenter
	{
		PVZoneID zone_id1;
		PVZoneID zone_id2;
		double factor1;
		double factor2;

		barycenter()
		{
			clear();
		}

		void clear()
		{
			zone_id1 = PVZONEID_INVALID;
			zone_id2 = PVZONEID_INVALID;
			factor1 = 0.0;
			factor2 = 0.0;
		}
	};

public:
	PVFullParallelViewSelectionRectangle(PVFullParallelScene* fps);

public:
	void clear() override;

public:
	void update_position();

protected:
	void commit(bool use_selection_modifiers) override;

	Picviz::PVView& lib_view() override;

private:
	void store();

	PVFullParallelScene* scene_parent();
	PVFullParallelScene const* scene_parent() const;

	PVLinesView& get_lines_view();
	PVLinesView const& get_lines_view() const;

private:
	PVFullParallelScene* _fps;
	barycenter           _barycenter;
};

}

#endif // PVPARALLELVIEW_PVFULLPARALLELVIEWSELECTIONSQUARE_H
