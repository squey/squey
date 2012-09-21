
#ifndef PVPARALLELVIEW_PVSELECTIONAXISSLIDERS_H
#define PVPARALLELVIEW_PVSELECTIONAXISSLIDERS_H

#include <pvkernel/core/PVAlgorithms.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/PVAbstractAxisSliders.h>
#include <pvparallelview/PVAxisSlider.h>
#include <pvparallelview/PVSlidersManager.h>

/* TODO: add a method to delete all the corresponding PVAxisSliders
 *       like a destroy() { hive::call(..., del_selection_sliders, _axis, _id); }
 */

namespace PVParallelView
{

class PVSelectionAxisSliders : public PVAbstractAxisSliders
{
Q_OBJECT

private:
	typedef PVSlidersManager::id_t id_t;

public:
	typedef std::pair<PVRow, PVRow> interval_t;

public:
	PVSelectionAxisSliders(QGraphicsItem *parent);

	void initialize(PVSlidersManager_p sm_p,
	                PVZoneID axis_index, id_t id,
	                uint32_t y_min, uint32_t y_max);

	interval_t get_range() const
	{
		PVRow v_min = _sl_min->value();
		PVRow v_max = _sl_max->value();

		return std::make_pair(PVCore::min(v_min, v_max),
		                      PVCore::max(v_min, v_max));
	}

	virtual bool is_moving() const
	{
		return (_sl_min->is_moving() || _sl_max->is_moving());
	}

signals:
	void sliders_moved();

private slots:
	void do_sliders_moved()
	{
		emit sliders_moved();
		PVHive::call<FUNC(PVSlidersManager::update_selection_sliders)>(_sliders_manager_p,
		                                                               _axis_index, _id,
		                                                               _sl_min->value(),
		                                                               _sl_max->value());
	}

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
	PVSlidersManager_p            _sliders_manager_p;
	selection_sliders_del_obs     _ssd_obs;
	selection_sliders_update_obs  _ssu_obs;
	PVAxisSlider                 *_sl_min;
	PVAxisSlider                 *_sl_max;
	PVZoneID                      _axis_index;
	id_t                          _id;
};

}

#endif // PVPARALLELVIEW_PVSELECTIONAXISSLIDERS_H
