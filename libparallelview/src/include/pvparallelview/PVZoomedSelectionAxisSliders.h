
#ifndef PVPARALLELVIEW_PVZOOMEDSELECTIONAXISSLIDERS_H
#define PVPARALLELVIEW_PVZOOMEDSELECTIONAXISSLIDERS_H

#include <pvkernel/core/PVAlgorithms.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/PVAbstractRangeAxisSliders.h>

/* TODO: add a method to delete all the corresponding PVAxisSliders
 *       like a destroy() { hive::call(..., del_selection_sliders, _axis, _id); }
 */

namespace PVParallelView
{

class PVSlidersGroup;

class PVZoomedSelectionAxisSliders : public PVAbstractRangeAxisSliders
{
Q_OBJECT

private:
	typedef PVSlidersManager::axis_id_t axis_id_t;
	typedef PVSlidersManager::id_t id_t;

public:
	typedef std::pair<PVRow, PVRow> range_t;

public:
	PVZoomedSelectionAxisSliders(QGraphicsItem *parent,
	                             PVSlidersManager_p sm_p,
	                             PVSlidersGroup *group);

	virtual void initialize(id_t id, int64_t y_min, int64_t y_max);

	id_t get_id() const
	{
		return _id;
	}

	void set_value(int64_t y_min, int64_t y_max);

public slots:
	virtual void remove_from_axis();

private slots:
	void do_sliders_moved();

private:
	class zoomed_selection_sliders_del_obs :
		public PVHive::PVFuncObserver<PVSlidersManager,
		                              FUNC(PVSlidersManager::del_zoomed_selection_sliders)>
	{
	public:
		zoomed_selection_sliders_del_obs(PVZoomedSelectionAxisSliders *parent) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const;

	private:
		PVZoomedSelectionAxisSliders *_parent;
	};

	class zoomed_selection_sliders_update_obs :
		public PVHive::PVFuncObserver<PVSlidersManager,
		                              FUNC(PVSlidersManager::update_zoomed_selection_sliders)>
	{
	public:
		zoomed_selection_sliders_update_obs(PVZoomedSelectionAxisSliders *parent) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const;

	private:
		PVZoomedSelectionAxisSliders *_parent;
	};

private:
	zoomed_selection_sliders_del_obs    _zssd_obs;
	zoomed_selection_sliders_update_obs _zssu_obs;
	id_t                                _id;
};

}

#endif // PVPARALLELVIEW_PVZOOMEDSELECTIONAXISSLIDERS_H
