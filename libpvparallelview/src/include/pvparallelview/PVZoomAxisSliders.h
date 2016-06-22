/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVZOOMAXISSLIDERS_H
#define PVPARALLELVIEW_PVZOOMAXISSLIDERS_H

#include <sigc++/sigc++.h>

#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVAbstractRangeAxisSliders.h>
#include <pvparallelview/PVSlidersManager.h>

namespace PVParallelView
{

class PVSlidersGroup;

class PVZoomAxisSliders : public PVAbstractRangeAxisSliders, public sigc::trackable
{
	Q_OBJECT

  private:
	typedef PVSlidersManager::axis_id_t axis_id_t;
	typedef PVSlidersManager::id_t id_t;

  public:
	PVZoomAxisSliders(QGraphicsItem* parent, PVSlidersManager* sm_p, PVSlidersGroup* group);

	void initialize(id_t id, int64_t y_min, int64_t y_max);

  public Q_SLOTS:
	void remove_from_axis() override;

  private Q_SLOTS:
	void do_sliders_moved();

  private:
	void on_zoom_sliders_update(axis_id_t axis_id,
	                            id_t id,
	                            int64_t y_min,
	                            int64_t y_max,
	                            PVSlidersManager::ZoomSliderChange change);

  private:
	id_t _id;
};
}

#endif // PVPARALLELVIEW_PVZOOMAXISSLIDERS_H
