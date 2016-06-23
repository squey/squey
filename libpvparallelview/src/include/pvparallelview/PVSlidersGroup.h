/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVSLIDERSGROUP_H
#define PVPARALLELVIEW_PVSLIDERSGROUP_H

#include <sigc++/sigc++.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVSlidersManager.h>
#include <pvparallelview/PVAbstractRangeAxisSliders.h>

#include <unordered_map>

#include <QObject>
#include <QGraphicsItemGroup>

namespace PVParallelView
{

class PVSelectionAxisSliders;
class PVZoomedSelectionAxisSliders;
class PVZoomAxisSliders;

class PVSlidersGroup : public QObject, public QGraphicsItemGroup, public sigc::trackable
{
	Q_OBJECT

  private:
	typedef PVSlidersManager::id_t id_t;
	typedef PVSlidersManager::range_geometry_t range_geometry_t;

  public:
	typedef PVSlidersManager::axis_id_t axis_id_t;
	typedef PVAbstractRangeAxisSliders::range_t range_t;
	typedef std::vector<range_t> selection_ranges_t;

  public:
	PVSlidersGroup(PVSlidersManager* sm_p,
	               const axis_id_t& axis_id,
	               QGraphicsItem* parent = nullptr);
	~PVSlidersGroup();

	void remove_selection_sliders();
	void remove_zoom_slider();

	void delete_own_selection_sliders();
	void delete_own_zoom_slider();

	void set_axis_id(const axis_id_t& axis_id) { _axis_id = axis_id; }

	const axis_id_t& get_axis_id() const { return _axis_id; }

	void set_axis_scale(float s);

	float get_axis_scale() const { return _axis_scale; }

	QRectF boundingRect() const override;

	void add_zoom_sliders(int64_t y_min, int64_t y_max);

	void add_selection_sliders(int64_t y_min, int64_t y_max);

	PVZoomedSelectionAxisSliders* add_zoomed_selection_sliders(int64_t y_min, int64_t y_max);

	bool sliders_moving() const;

	selection_ranges_t get_selection_ranges() const;

  Q_SIGNALS:
	void selection_sliders_moved(const axis_id_t axis_id);

  protected Q_SLOTS:
	void selection_slider_moved() { Q_EMIT selection_sliders_moved(get_axis_id()); }

  private:
	/**
	 * Initialize and insert a new sliders pair
	 *
	 * If sliders is nullptr, it is created.
	 * If id is 0, it is deduced from sliders.
	 */
	void add_new_zoom_sliders(id_t id, int64_t y_min, int64_t y_max);
	void add_new_selection_sliders(PVSelectionAxisSliders* sliders,
	                               id_t id,
	                               int64_t y_min,
	                               int64_t y_max);
	void add_new_zoomed_selection_sliders(PVZoomedSelectionAxisSliders* sliders,
	                                      id_t id,
	                                      int64_t y_min,
	                                      int64_t y_max);

	void del_zoom_sliders(id_t id);
	void del_selection_sliders(id_t id);
	void del_zoomed_selection_sliders(id_t id);

  private:
	void
	on_new_zoom_slider(axis_id_t axis_d, PVSlidersManager::id_t id, int64_t y_min, int64_t y_max);
	void on_new_selection_sliders(axis_id_t axis_d,
	                              PVSlidersManager::id_t id,
	                              int64_t y_min,
	                              int64_t y_max);
	void on_new_zoomed_selection_sliders(axis_id_t axis_d,
	                                     PVSlidersManager::id_t id,
	                                     int64_t y_min,
	                                     int64_t y_max);
	void on_del_zoom_sliders(axis_id_t axis_d, PVSlidersManager::id_t id);
	void on_del_selection_sliders(axis_id_t axis_d, PVSlidersManager::id_t id);
	void on_del_zoomed_selection_sliders(axis_id_t axis_d, PVSlidersManager::id_t id);

  private:
	typedef std::unordered_map<id_t, PVSelectionAxisSliders*> sas_set_t;
	typedef std::unordered_map<id_t, PVZoomedSelectionAxisSliders*> zsas_set_t;
	typedef std::unordered_map<id_t, PVZoomAxisSliders*> zas_set_t;

  private:
	PVSlidersManager* _sliders_manager_p;
	axis_id_t _axis_id;
	float _axis_scale;

	sas_set_t _selection_sliders;
	zsas_set_t _zoomed_selection_sliders;
	zas_set_t _zoom_sliders;
};
}

#endif // PVPARALLELVIEW_PVSLIDERSGROUP_H
