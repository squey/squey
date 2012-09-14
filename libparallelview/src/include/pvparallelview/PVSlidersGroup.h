
#ifndef PVPARALLELVIEW_PVSLIDERSGROUP_H
#define PVPARALLELVIEW_PVSLIDERSGROUP_H

#include <pvhive/PVHive.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVSlidersManager.h>
#include <pvparallelview/PVAxisSlider.h>

#include <QObject>
#include <QGraphicsItemGroup>

namespace PVParallelView
{

typedef std::pair<PVAxisSlider*, PVAxisSlider*> PVAxisRangeSliders;

class PVSlidersGroup : public QObject, public QGraphicsItemGroup
{
	Q_OBJECT

public:
	typedef std::vector<std::pair<PVRow, PVRow> > selection_ranges_t;

public:
	PVSlidersGroup(PVSlidersManager_p sm_p, PVCol axis_index, QGraphicsItem *parent = nullptr);

	void add_zoom_sliders(uint32_t y_min, uint32_t y_max);

	void add_selection_sliders(uint32_t y_min, uint32_t y_max);

	bool sliders_moving() const;

	selection_ranges_t get_selection_ranges() const;

signals:
	void selection_sliders_moved(PVZoneID);

protected slots:
	void selection_slider_moved() { emit selection_sliders_moved(_axis_index); }

private:
	class zoom_sliders_new_obs :
		public PVHive::PVFuncObserverSignal<PVSlidersManager,
		                                    FUNC(PVSlidersManager::new_zoom_sliders)>
	{
	public:
		zoom_sliders_new_obs(PVSlidersGroup *parent) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const;

	private:
		PVSlidersGroup *_parent;
	};

private:
	PVSlidersManager_p   _sliders_manager_p;
	zoom_sliders_new_obs _zsn_obs;
	PVCol                _axis_index;

	std::vector<PVAxisRangeSliders> _all_sliders;
	std::vector<PVAxisRangeSliders> _selection_sliders;
};

}

#endif // PVPARALLELVIEW_PVSLIDERSGROUP_H
