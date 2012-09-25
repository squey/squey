
#ifndef PVPARALLELVIEW_PVSELECTIONAXISSLIDERS_H
#define PVPARALLELVIEW_PVSELECTIONAXISSLIDERS_H

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

class PVSelectionAxisSliders : public PVAbstractRangeAxisSliders
{
Q_OBJECT

private:
	typedef PVSlidersManager::axe_id_t axe_id_t;
	typedef PVSlidersManager::id_t id_t;

public:
	typedef std::pair<PVRow, PVRow> range_t;

public:
	PVSelectionAxisSliders(QGraphicsItem *parent,
	                       PVSlidersManager_p sm_p,
	                       PVSlidersGroup *group);

	virtual void initialize(id_t id, uint32_t y_min, uint32_t y_max);

	id_t get_id() const
	{
		return _id;
	}

private slots:
	void do_sliders_moved();

private:
	class selection_sliders_del_obs :
		public PVHive::PVFuncObserver<PVSlidersManager,
		                              FUNC(PVSlidersManager::del_selection_sliders)>
	{
	public:
		selection_sliders_del_obs(PVSelectionAxisSliders *parent) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const;

	private:
		PVSelectionAxisSliders *_parent;
	};

	class selection_sliders_update_obs :
		public PVHive::PVFuncObserver<PVSlidersManager,
		                              FUNC(PVSlidersManager::update_selection_sliders)>
	{
	public:
		selection_sliders_update_obs(PVSelectionAxisSliders *parent) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const;

	private:
		PVSelectionAxisSliders *_parent;
	};

private:
	selection_sliders_del_obs     _ssd_obs;
	selection_sliders_update_obs  _ssu_obs;
	id_t                          _id;
};

}

#endif // PVPARALLELVIEW_PVSELECTIONAXISSLIDERS_H
