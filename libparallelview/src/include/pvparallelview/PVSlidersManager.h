
#ifndef PVPARALLELVIEW_PVSLIDERSMANAGER_H
#define PVPARALLELVIEW_PVSLIDERSMANAGER_H

#include <pvbase/types.h>

#include <pvkernel/core/PVSharedPointer.h>

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
	typedef void* id_t;

	struct interval_geometry_t
	{
		uint32_t y_min;
		uint32_t y_max;
	};

	typedef std::function<void(const PVCol, const id_t,
	                           const interval_geometry_t &)> interval_functor_t;

public:
	PVSlidersManager();

public:
	~PVSlidersManager();

public:
	/**
	 * Function to observe (in PVHive way) to be notified when a new
	 * interval slider is added
	 *
	 * @param axis the axis the slider is associated with
	 * @param id the id the slider is associated with
	 * @param y_min the low position of the sliders
	 * @param y_max the high position of the sliders
	 */
	void new_selection_sliders(const PVCol axis, const id_t id,
	                           const uint32_t y_min, const uint32_t y_max);
	void new_zoom_sliders(const PVCol axis, const id_t id,
	                      const uint32_t y_min, const uint32_t y_max);

	/**
	 * Function to observe (in PVHive way) to be notified when a new
	 * interval slider is deleted
	 *
	 * @param axis the axis the slider is associated with
	 * @param id the id the slider is associated with
	 */
	void del_selection_sliders(const PVCol axis, const id_t id);
	void del_zoom_sliders(const PVCol axis, const id_t id);

	/**
	 * Function to observe (in PVHive way) to be notified when a
	 * interval slider is changed
	 *
	 * @param axis the axis the slider is associated with
	 * @param id the id the slider is associated with
	 * @param y_min the low position of the sliders
	 * @param y_max the high position of the sliders
	 */
	void update_selection_sliders(const PVCol axis, const id_t id,
	                              const uint32_t y_min, const uint32_t y_max);
	void update_zoom_sliders(const PVCol axis, const id_t id,
	                         const uint32_t y_min, const uint32_t y_max);

	/**
	 * Function to iterate on all interval sliders
	 *
	 * @param functor the function called on each interval sliders
	 *
	 * This method has to be used when creating a new graphical view
	 * to initialize it with each existing interval sliders
	 */
	void iterate_selection_sliders(const interval_functor_t &functor) const;
	void iterate_zoom_sliders(const interval_functor_t &functor) const;

private:
	typedef std::map<id_t, interval_geometry_t> interval_geometry_list_t;
	typedef std::map<PVCol, interval_geometry_list_t> interval_geometry_set_t;

private:
	void new_interval_sliders(interval_geometry_set_t &interval,
	                          const PVCol axis, const id_t id,
	                          const uint32_t y_min, const uint32_t y_max);

	void del_interval_sliders(interval_geometry_set_t &interval,
	                          const PVCol axis, const id_t id);

	void update_interval_sliders(interval_geometry_set_t &interval,
	                             const PVCol axis, const id_t id,
	                             const uint32_t y_min, const uint32_t y_max);

	void iterate_interval_sliders(const interval_geometry_set_t &interval,
	                              const interval_functor_t &functor) const;

private:
	interval_geometry_set_t _zoom_geometries;
	interval_geometry_set_t _selection_geometries;
};

typedef PVCore::PVSharedPtr<PVSlidersManager> PVSlidersManager_p;

}

#endif // PVPARALLELVIEW_PVSLIDERSMANAGER_H
