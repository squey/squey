
#ifndef PVPARALLELVIEW_PVZOOMAXISSLIDERS_H
#define PVPARALLELVIEW_PVZOOMAXISSLIDERS_H

#include <pvkernel/core/PVAlgorithms.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/PVAbstractRangeAxisSliders.h>

namespace PVParallelView
{

class PVSlidersGroup;

class PVZoomAxisSliders : public PVAbstractRangeAxisSliders
{
	Q_OBJECT

private:
	typedef PVSlidersManager::axe_id_t axe_id_t;
	typedef PVSlidersManager::id_t id_t;

public:
	typedef std::pair<PVRow, PVRow> range_t;

public:
	PVZoomAxisSliders(QGraphicsItem *parent,
	                  PVSlidersManager_p sm_p,
	                  PVSlidersGroup *group);

	void initialize(id_t id, uint32_t y_min, uint32_t y_max);

public slots:
	virtual void remove_from_axis();

private slots:
	void do_sliders_moved();

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
	zoom_sliders_del_obs     _zsd_obs;
	zoom_sliders_update_obs  _zsu_obs;
	id_t                     _id;
};

}

#endif // PVPARALLELVIEW_PVZOOMAXISSLIDERS_H
