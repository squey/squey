
#ifndef PVPARALLELVIEW_PVSLIDERSGROUP_H
#define PVPARALLELVIEW_PVSLIDERSGROUP_H

#include <pvhive/PVHive.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVSlidersManager.h>
#include <pvparallelview/PVAxisSlider.h>
#include <pvparallelview/PVAbstractAxisSliders.h>
#include <pvparallelview/PVZoomAxisSliders.h>
#include <pvparallelview/PVSelectionAxisSliders.h>

#include <vector>
#include <set>

#include <QObject>
#include <QGraphicsItemGroup>

namespace PVParallelView
{

class PVSlidersGroup : public QObject, public QGraphicsItemGroup
{
	Q_OBJECT

private:
	typedef PVSlidersManager::id_t id_t;
	typedef PVSlidersManager::interval_geometry_t interval_geometry_t;

public:
	typedef std::vector<std::pair<PVRow, PVRow> > selection_ranges_t;

public:
	PVSlidersGroup(PVSlidersManager_p sm_p, PVCol axis_index, QGraphicsItem *parent = nullptr);
	~PVSlidersGroup();

	QRectF boundingRect() const
	{
		// TODO: the width depend of the children's width
		return QRectF(-10, 0, 10, 1024);
	}

	void add_zoom_sliders(uint32_t y_min, uint32_t y_max);

	void add_selection_sliders(uint32_t y_min, uint32_t y_max);

	bool sliders_moving() const;

	selection_ranges_t get_selection_ranges() const;

signals:
	void selection_sliders_moved(PVZoneID);

protected slots:
	void selection_slider_moved() { emit selection_sliders_moved(_axis_index); }

private:
	/**
	 * Initialize and insert a new sliders pair
	 *
	 * If sliders is nullptr, it is created.
	 * If id is 0, it is deduced from sliders.
	 */
	void add_new_zoom_sliders(PVCol axis, id_t id,
	                          uint32_t y_min, uint32_t y_max);
	void add_new_selection_sliders(PVSelectionAxisSliders* sliders,
	                               PVCol axis, id_t id,
	                               uint32_t y_min, uint32_t y_max);

private:
	class zoom_sliders_new_obs :
		public PVHive::PVFuncObserver<PVSlidersManager,
		                              FUNC(PVSlidersManager::new_zoom_sliders)>
	{
	public:
		zoom_sliders_new_obs(PVSlidersGroup *parent) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const;

	private:
		PVSlidersGroup *_parent;
	};

	class selection_sliders_new_obs :
		public PVHive::PVFuncObserver<PVSlidersManager,
		                              FUNC(PVSlidersManager::new_selection_sliders)>
	{
	public:
		selection_sliders_new_obs(PVSlidersGroup *parent) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const;

	private:
		PVSlidersGroup *_parent;
	};

	class zoom_sliders_del_obs :
		public PVHive::PVFuncObserver<PVSlidersManager,
		                              FUNC(PVSlidersManager::del_zoom_sliders)>
	{
	public:
		zoom_sliders_del_obs(PVSlidersGroup *parent) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const;

	private:
		PVSlidersGroup *_parent;
	};

	class selection_sliders_del_obs :
		public PVHive::PVFuncObserver<PVSlidersManager,
		                              FUNC(PVSlidersManager::del_selection_sliders)>
	{
	public:
		selection_sliders_del_obs(PVSlidersGroup *parent) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const;

	private:
		PVSlidersGroup *_parent;
	};

private:
	typedef std::vector<PVAbstractAxisSliders*>  aas_set_t;
	typedef std::vector<PVSelectionAxisSliders*> sas_set_t;
	typedef std::set<id_t>                       id_set_t;

private:
	PVSlidersManager_p        _sliders_manager_p;
	zoom_sliders_new_obs      _zsn_obs;
	selection_sliders_new_obs _ssn_obs;
	zoom_sliders_del_obs      _zsd_obs;
	selection_sliders_del_obs _ssd_obs;
	PVCol                     _axis_index;

	aas_set_t                 _all_sliders;
	sas_set_t                 _selection_sliders;
	id_set_t                  _registered_ids;
};

}

#endif // PVPARALLELVIEW_PVSLIDERSGROUP_H
