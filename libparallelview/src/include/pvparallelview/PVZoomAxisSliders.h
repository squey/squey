
#ifndef PVPARALLELVIEW_PVZOOMAXISSLIDERS_H
#define PVPARALLELVIEW_PVZOOMAXISSLIDERS_H

#include <pvkernel/core/PVAlgorithms.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/PVAbstractAxisSliders.h>
#include <pvparallelview/PVAxisSlider.h>
#include <pvparallelview/PVSlidersManager.h>

namespace PVParallelView
{

class PVZoomAxisSliders : public PVAbstractAxisSliders
{
	Q_OBJECT

	private:
	typedef PVSlidersManager::id_t id_t;

public:
	typedef std::pair<PVRow, PVRow> interval_t;

public:
	PVZoomAxisSliders(QGraphicsItem *parent);

	void initialize(PVSlidersManager_p sm_p,
	                PVCol axis, id_t id,
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

		int change = PVParallelView::PVSlidersManager::ZoomSliderNone;

		if (_sl_min->is_moving()) {
			change = PVParallelView::PVSlidersManager::ZoomSliderMin;
		}

		if (_sl_max->is_moving()) {
			change |= PVParallelView::PVSlidersManager::ZoomSliderMax;
		}

		PVHive::call<FUNC(PVSlidersManager::update_zoom_sliders)>(_sliders_manager_p,
		                                                          _axis, _id,
		                                                          _sl_min->value(),
		                                                          _sl_max->value(),
		                                                          (PVParallelView::PVSlidersManager::ZoomSliderChange)change);
	}

private:
	class zoom_sliders_del_obs :
		public PVHive::PVFuncObserver<PVSlidersManager,
		                              FUNC(PVSlidersManager::del_zoom_sliders)>
	{
	public:
		zoom_sliders_del_obs(PVZoomAxisSliders *parent) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const;

	private:
		PVZoomAxisSliders *_parent;
	};

	class zoom_sliders_update_obs :
		public PVHive::PVFuncObserver<PVSlidersManager,
		                              FUNC(PVSlidersManager::update_zoom_sliders)>
	{
	public:
		zoom_sliders_update_obs(PVZoomAxisSliders *parent) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const;

	private:
		PVZoomAxisSliders *_parent;
	};

private:
	PVSlidersManager_p       _sliders_manager_p;
	zoom_sliders_del_obs     _zsd_obs;
	zoom_sliders_update_obs  _zsu_obs;
	PVAxisSlider            *_sl_min;
	PVAxisSlider            *_sl_max;
	PVCol                    _axis;
	id_t                     _id;
};

}

#endif // PVPARALLELVIEW_PVZOOMAXISSLIDERS_H
