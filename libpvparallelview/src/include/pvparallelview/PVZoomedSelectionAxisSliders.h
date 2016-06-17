/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVZOOMEDSELECTIONAXISSLIDERS_H
#define PVPARALLELVIEW_PVZOOMEDSELECTIONAXISSLIDERS_H

#include <sigc++/sigc++.h>

#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVAbstractRangeAxisSliders.h>

namespace PVParallelView
{

class PVSlidersGroup;

class PVZoomedSelectionAxisSliders : public PVAbstractRangeAxisSliders, public sigc::trackable
{
	Q_OBJECT

  private:
	typedef PVSlidersManager::axis_id_t axis_id_t;
	typedef PVSlidersManager::id_t id_t;

  public:
	PVZoomedSelectionAxisSliders(QGraphicsItem* parent,
	                             PVSlidersManager_p sm_p,
	                             PVSlidersGroup* group);

	virtual void initialize(id_t id, int64_t y_min, int64_t y_max);

	id_t get_id() const { return _id; }

	void set_value(int64_t y_min, int64_t y_max);

  public Q_SLOTS:
	void remove_from_axis() override;

  private Q_SLOTS:
	void do_sliders_moved();

  private:
	void on_zoomed_selection_sliders_update(axis_id_t axis_id,
	                                        PVSlidersManager::id_t id,
	                                        int64_t y_min,
	                                        int64_t y_max);

  private:
	id_t _id;
};
}

#endif // PVPARALLELVIEW_PVZOOMEDSELECTIONAXISSLIDERS_H
