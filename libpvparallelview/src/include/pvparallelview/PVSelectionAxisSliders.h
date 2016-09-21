/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVSELECTIONAXISSLIDERS_H
#define PVPARALLELVIEW_PVSELECTIONAXISSLIDERS_H

#include <sigc++/sigc++.h>

#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVAbstractRangeAxisSliders.h>

namespace PVParallelView
{

class PVSlidersGroup;

class PVSelectionAxisSliders : public PVAbstractRangeAxisSliders, public sigc::trackable
{
	Q_OBJECT

  private:
	typedef PVSlidersManager::id_t id_t;

  public:
	PVSelectionAxisSliders(QGraphicsItem* parent, PVSlidersManager* sm_p, PVSlidersGroup* group);

	void initialize(id_t id, int64_t y_min, int64_t y_max) override;

	id_t get_id() const { return _id; }

  public Q_SLOTS:
	void remove_from_axis() override;

  private Q_SLOTS:
	void do_sliders_moved();

  private:
	void on_selection_sliders_update(PVCol nraw_col, id_t id, int64_t y_min, int64_t y_max);

  private:
	id_t _id;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVSELECTIONAXISSLIDERS_H
