/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVPARALLELVIEW_PVSLIDERSMANAGER_H
#define PVPARALLELVIEW_PVSLIDERSMANAGER_H

#include <pvbase/types.h>

#include <sigc++/sigc++.h>

#include <pvparallelview/common.h>

#include <map>
#include <functional>

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
		ZoomSliderNone = 0,
		ZoomSliderMin = 1,
		ZoomSliderMax = 2,
		ZoomSliderBoth = 3 // 1 + 2
	} ZoomSliderChange;

	typedef void* id_t;

	struct range_geometry_t {
		int64_t y_min;
		int64_t y_max;
	};

	typedef std::function<void(const PVCombCol, const id_t, const range_geometry_t&)>
	    range_functor_t;

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
	void
	new_selection_sliders(PVCombCol col, const id_t id, const int64_t y_min, const int64_t y_max);
	void new_zoom_sliders(PVCombCol col, const id_t id, const int64_t y_min, const int64_t y_max);
	void new_zoomed_selection_sliders(PVCombCol col,
	                                  const id_t id,
	                                  const int64_t y_min,
	                                  const int64_t y_max);

	/**
	 * Function to observe (in PVHive way) to be notified when a new
	 * range sliders pair is deleted
	 *
	 * @param axis the axis the slider is associated with
	 * @param id the id the slider is associated with
	 */
	void del_selection_sliders(PVCombCol col, const id_t id);
	void del_zoom_sliders(PVCombCol col, const id_t id);
	void del_zoomed_selection_sliders(PVCombCol col, const id_t id);

	/**
	 * Function to observe (in PVHive way) to be notified when a
	 * range sliders pair is changed
	 *
	 * @param axis the axis the slider is associated with
	 * @param id the id the slider is associated with
	 * @param y_min the low position of the sliders
	 * @param y_max the high position of the sliders
	 */
	void update_selection_sliders(PVCombCol col,
	                              const id_t id,
	                              const int64_t y_min,
	                              const int64_t y_max);
	void update_zoom_sliders(PVCombCol col,
	                         const id_t id,
	                         const int64_t y_min,
	                         const int64_t y_max,
	                         const ZoomSliderChange change);
	void update_zoomed_selection_sliders(PVCombCol col,
	                                     const id_t id,
	                                     const int64_t y_min,
	                                     const int64_t y_max);

	/**
	 * Function to iterate on all range sliders
	 *
	 * @param functor the function called on each range sliders pair
	 *
	 * This method has to be used when creating a new graphical view
	 * to initialize it with each existing range sliders
	 */
	void iterate_selection_sliders(const range_functor_t& functor) const;
	void iterate_zoom_sliders(const range_functor_t& functor) const;
	void iterate_zoomed_selection_sliders(const range_functor_t& functor) const;

  private:
	typedef std::map<id_t, range_geometry_t> range_geometry_list_t;
	typedef std::map<PVCombCol, range_geometry_list_t> range_geometry_set_t;

  private:
	void new_range_sliders(range_geometry_set_t& range,
	                       PVCombCol col,
	                       const id_t id,
	                       const int64_t y_min,
	                       const int64_t y_max);

	void del_range_sliders(range_geometry_set_t& range, PVCombCol col, const id_t id);

	void update_range_sliders(range_geometry_set_t& range,
	                          PVCombCol col,
	                          const id_t id,
	                          const int64_t y_min,
	                          const int64_t y_max);

	void iterate_range_sliders(const range_geometry_set_t& range,
	                           const range_functor_t& functor) const;

  public:
	sigc::signal<void, PVCombCol, id_t, int64_t, int64_t> _new_zoom_sliders;
	sigc::signal<void, PVCombCol, id_t, int64_t, int64_t> _new_selection_sliders;
	sigc::signal<void, PVCombCol, id_t, int64_t, int64_t> _new_zoomed_selection_sliders;
	sigc::signal<void, PVCombCol, id_t> _del_zoom_sliders;
	sigc::signal<void, PVCombCol, id_t> _del_selection_sliders;
	sigc::signal<void, PVCombCol, id_t> _del_zoomed_selection_sliders;
	sigc::signal<void, PVCombCol, id_t, int64_t, int64_t, ZoomSliderChange> _update_zoom_sliders;
	sigc::signal<void, PVCombCol, id_t, int64_t, int64_t> _update_selection_sliders;
	sigc::signal<void, PVCombCol, id_t, int64_t, int64_t> _update_zoomed_selection_sliders;

  private:
	range_geometry_set_t _zoom_geometries;
	range_geometry_set_t _selection_geometries;
	range_geometry_set_t _zoomed_selection_geometries;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVSLIDERSMANAGER_H
