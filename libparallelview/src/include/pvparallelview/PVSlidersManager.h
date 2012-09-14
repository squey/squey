
#ifndef PVPARALLELVIEW_PVSLIDERSMANAGER_H
#define PVPARALLELVIEW_PVSLIDERSMANAGER_H

#include <pvbase/types.h>

#include <pvkernel/core/PVSharedPointer.h>

#include <map>

namespace PVParallelView
{

class PVSlidersManager
{
public:
	typedef void* id_t;

	struct interval_geometry_t
	{
		uint32_t y_min;
		uint32_t y_max;
	};

	typedef std::function<void(const PVCol, const id_t,
	                           const interval_geometry_t &)> zoom_functor_t;

public:
	PVSlidersManager();

public:
	~PVSlidersManager();

public:
	/**
	 * Function to observe (in PVHive way) to be notified when a new zoom
	 * slider is added
	 *
	 * @param axis the axis the slider is associated with
	 * @param id the id the slider is associated with
	 * @param y_min the low position of the slider
	 * @param y_max the high position of the slider
	 */
	void new_zoom_sliders(const PVCol axis, const id_t id,
	                      const uint32_t y_min, const uint32_t y_max);

	/**
	 * Function to observe (in PVHive way) to be notified when a new zoom
	 * slider is deleted
	 *
	 * @param axis the axis the slider is associated with
	 * @param id the id the slider is associated with
	 */
	void del_zoom_sliders(const PVCol axis, const id_t id);

	/**
	 * Function to observe (in PVHive way) to be notified when a zoom
	 * slider is changed
	 *
	 * @param axis the axis the slider is associated with
	 * @param id the id the slider is associated with
	 * @param y_min the low position of the slider
	 * @param y_max the high position of the slider
	 */
	void update_zoom_sliders(const PVCol axis, const id_t id,
	                         const uint32_t y_min, const uint32_t y_max);

	/**
	 * Function to iterate on all zoom sliders
	 *
	 * @param functor the function called on each zoom sliders
	 *
	 * This method has to be used when creating a new graphical view
	 * to initialize it with each existing zoom sliders
	 */
	void iterate_zoom_sliders(const zoom_functor_t &functor) const;

private:
	typedef std::map<id_t, interval_geometry_t> interval_geometry_list_t;
	typedef std::map<PVCol, interval_geometry_list_t> interval_geometry_set_t;

private:
	interval_geometry_set_t _zoom_geometries;
};

typedef PVCore::PVSharedPtr<PVSlidersManager> PVSlidersManager_p;

}

#endif // PVPARALLELVIEW_PVSLIDERSMANAGER_H
