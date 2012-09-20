
#ifndef PVPARALLELVIEW_PVSLIDERSMANAGER_H
#define PVPARALLELVIEW_PVSLIDERSMANAGER_H

#include <pvbase/types.h>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvparallelview/common.h>

#include <map>

/* TODO: test if FuncObserver work with inline functions
 *
 * IDEA: allow to highlight a distant slider?
 *
 * IDEA: allow to hide/show a distant slider?
 */

namespace PVParallelView
{

class PVSlidersManager
{
public:
	typedef enum {
		ZoomSliderNone   = 0,
		ZoomSliderMin    = 1,
		ZoomSliderMax    = 2,
		ZoomSliderBoth   = 3 // 1 + 2
	} ZoomSliderChange;

	typedef void* id_t;

	struct range_geometry_t
	{
		uint32_t y_min;
		uint32_t y_max;
	};

	typedef std::function<void(const PVZoneID, const id_t,
	                           const range_geometry_t &)> range_functor_t;

public:
	PVSlidersManager();

public:
	~PVSlidersManager();

public:
	/**
	 * Function to observe (in PVHive way) to be notified when a new
	 * range sliders pair is added
	 *
	 * @param axis the axis the slider is associated with
	 * @param id the id the slider is associated with
	 * @param y_min the low position of the sliders
	 * @param y_max the high position of the sliders
	 */
	void new_selection_sliders(const PVZoneID axis_index, const id_t id,
	                           const uint32_t y_min, const uint32_t y_max);
	void new_zoom_sliders(const PVZoneID axis_index, const id_t id,
	                      const uint32_t y_min, const uint32_t y_max);

	/**
	 * Function to observe (in PVHive way) to be notified when a new
	 * range sliders pair is deleted
	 *
	 * @param axis the axis the slider is associated with
	 * @param id the id the slider is associated with
	 */
	void del_selection_sliders(const PVZoneID axis_index, const id_t id);
	void del_zoom_sliders(const PVZoneID axis_index, const id_t id);

	/**
	 * Function to observe (in PVHive way) to be notified when a
	 * range sliders pair is changed
	 *
	 * @param axis the axis the slider is associated with
	 * @param id the id the slider is associated with
	 * @param y_min the low position of the sliders
	 * @param y_max the high position of the sliders
	 */
	void update_selection_sliders(const PVZoneID axis_index, const id_t id,
	                              const uint32_t y_min, const uint32_t y_max);
	void update_zoom_sliders(const PVZoneID axis_index, const id_t id,
	                         const uint32_t y_min, const uint32_t y_max,
	                         const ZoomSliderChange change);

	/**
	 * Function to iterate on all range sliders
	 *
	 * @param functor the function called on each range sliders pair
	 *
	 * This method has to be used when creating a new graphical view
	 * to initialize it with each existing range sliders
	 */
	void iterate_selection_sliders(const range_functor_t &functor) const;
	void iterate_zoom_sliders(const range_functor_t &functor) const;

private:
	typedef std::map<id_t, range_geometry_t> range_geometry_list_t;
	typedef std::map<PVZoneID, range_geometry_list_t> range_geometry_set_t;

private:
	void new_range_sliders(range_geometry_set_t &range,
	                          const PVZoneID axis_index, const id_t id,
	                          const uint32_t y_min, const uint32_t y_max);

	void del_range_sliders(range_geometry_set_t &range,
	                          const PVZoneID axis_index, const id_t id);

	void update_range_sliders(range_geometry_set_t &range,
	                             const PVZoneID axis_index, const id_t id,
	                             const uint32_t y_min, const uint32_t y_max);

	void iterate_range_sliders(const range_geometry_set_t &range,
	                              const range_functor_t &functor) const;

private:
	range_geometry_set_t _zoom_geometries;
	range_geometry_set_t _selection_geometries;
};

typedef PVCore::PVSharedPtr<PVSlidersManager> PVSlidersManager_p;

}

#endif // PVPARALLELVIEW_PVSLIDERSMANAGER_H
