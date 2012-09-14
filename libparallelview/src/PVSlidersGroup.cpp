
#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVSlidersGroup.h>
#include <pvparallelview/PVAxisSlider.h>

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::PVSlidersGroup
 *****************************************************************************/

PVParallelView::PVSlidersGroup::PVSlidersGroup(PVSlidersManager_p sm_p,
                                               PVCol axis_index,
                                               QGraphicsItem *parent) :
	QGraphicsItemGroup(parent),
	_sliders_manager_p(sm_p),
	_zsn_obs(this),
	_axis_index(axis_index)
{
	// does not care about children events
	// RH: this method is obsolete in Qt 4.8 and replaced with
	//     setFiltersChildEvents() but it does not have the same
	//     behaviour... so I keep setHandlesChildEvents()
	setHandlesChildEvents(false);

	PVHive::PVHive::get().register_func_observer(_sliders_manager_p,
	                                             _zsn_obs);
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_zoom_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_zoom_sliders(uint32_t y_min,
                                                      uint32_t y_max)
{
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::add_selection_sliders
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::add_selection_sliders(uint32_t y_min,
                                                           uint32_t y_max)
{
	PVParallelView::PVAxisRangeSliders sliders;

	sliders.first = new PVParallelView::PVAxisSlider(0, PVParallelView::ImageHeight, y_min);
	sliders.second = new PVParallelView::PVAxisSlider(0, PVParallelView::ImageHeight, y_max);

	addToGroup(sliders.first);
	addToGroup(sliders.second);

	sliders.first->setPos(pos());
	sliders.second->setPos(pos());

	_all_sliders.push_back(sliders);
	_selection_sliders.push_back(sliders);

	// Connection
	connect(sliders.first, SIGNAL(slider_moved()), this, SLOT(selection_slider_moved()));
	connect(sliders.second, SIGNAL(slider_moved()), this, SLOT(selection_slider_moved()));
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::get_selection_ranges
 *****************************************************************************/

PVParallelView::PVSlidersGroup::selection_ranges_t
PVParallelView::PVSlidersGroup::get_selection_ranges() const
{
	selection_ranges_t ranges;

	for (PVParallelView::PVAxisRangeSliders sliders : _selection_sliders) {
		PVRow v_min = PVCore::min(sliders.first->value(), sliders.second->value());
		PVRow v_max = PVCore::max(sliders.first->value(), sliders.second->value());
		ranges.push_back(std::make_pair(v_min, v_max));
	}

	return ranges;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::sliders_moving
 *****************************************************************************/

bool PVParallelView::PVSlidersGroup::sliders_moving() const
{
	for (PVParallelView::PVAxisRangeSliders sliders : _all_sliders) {
		if (sliders.first->is_moving() || sliders.second->is_moving()) {
			return true;
		}
	}
	return false;
}

/*****************************************************************************
 * PVParallelView::PVSlidersGroup::zoom_sliders_new_obs::update
 *****************************************************************************/

void PVParallelView::PVSlidersGroup::zoom_sliders_new_obs::update(arguments_deep_copy_type const& args) const
{
	PVCol axis = std::get<0>(args);

	if (axis == _parent->_axis_index) {
		PVSlidersManager::id_t id = std::get<1>(args);
		uint32_t y_min = std::get<2>(args);
		uint32_t y_max = std::get<3>(args);
		printf("##### PVSlidersGroup::zoom_sliders_new_obs: add new zoom sliders: %d %p %u %u\n",
		       axis, id, y_min, y_max);
	}
}
