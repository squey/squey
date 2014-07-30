/**
 * \file PVAxisGraphicsItem.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVAXISGRAPHICSITEM_H
#define PVPARALLELVIEW_PVAXISGRAPHICSITEM_H

#include <iostream>
#include <vector>
#include <utility>

#include <QGraphicsItem>
class QPropertyAnimation;

#include <pvkernel/core/PVAlgorithms.h>

#include <picviz/PVAxis.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVCallHelper.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVSlidersManager.h>
#include <pvparallelview/PVSlidersGroup.h>

class QGraphicsSimpleTextItem;

namespace PVParallelView
{

class PVAxisHeader;

namespace __impl
{
class PVToolTipEventFilter;
class PVAxisSelectedAnimation;
}

class PVAxisLabel;
class PVFullParallelScene;

class PVAxisGraphicsItem : public QObject, public QGraphicsItemGroup
{
	Q_OBJECT

	friend class __impl::PVToolTipEventFilter;
	friend class __impl::PVAxisSelectedAnimation;
	friend class PVAxisHeader;

public:
	static constexpr qreal label_rotation = -45.;

public:
	typedef PVSlidersGroup::selection_ranges_t selection_ranges_t;
	typedef PVSlidersManager::axis_id_t        axis_id_t;

	// Used to draw the axis out of the image zone
	constexpr static int axis_extend = 8;

public:
	PVAxisGraphicsItem(PVSlidersManager_p sm_p, Picviz::PVView const& view, const axis_id_t &axis_id);
	~PVAxisGraphicsItem();

	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0) override;
	QRectF boundingRect() const override;

	void update_axis_label_info();
	void update_axis_min_max_info();
	void update_layer_min_max_info();

	void set_min_max_visible(const bool visible);

	PVSlidersGroup *get_sliders_group()
	{
		return _sliders_group;
	}

	const PVSlidersGroup *get_sliders_group() const
	{
		return _sliders_group;
	}

	const axis_id_t get_axis_id() const
	{
		return _axis_id;
	}

	PVCol get_combined_axis_column() const
	{
		return _lib_view.axes_combination.get_index_by_id(_axis_id);
	}

	PVCol get_original_axis_column() const
	{
		return _lib_view.axes_combination.get_axis_column_index(_lib_view.axes_combination.get_index_by_id(_axis_id));
	}

	QRectF get_top_decoration_scene_bbox() const;
	QRectF get_bottom_decoration_scene_bbox() const;

	void set_axis_length(int l);

	void set_zone_width(int w);

	QRect map_from_scene(QRectF rect) const
	{
		QPointF point = mapFromScene(rect.topLeft());
		return QRect(point.x(), point.y(), rect.width(), rect.height());
	}

	selection_ranges_t get_selection_ranges() const
	{
		return get_sliders_group()->get_selection_ranges();
	}

	bool is_last_axis() const;

	void highlight(bool start);

	PVAxisLabel* label() const { return _label; }

public slots:
	void emit_new_zoomed_parallel_view(int axis_id)
	{
		emit new_zoomed_parallel_view(axis_id);
	}

protected:
	void show_tooltip(QGraphicsTextItem* gti, QGraphicsSceneHelpEvent* event) const;

signals:
	void new_zoomed_parallel_view(int axis_id);
	void mouse_hover_entered(PVCol axis, bool entered);
	void mouse_clicked(PVCol axis);

private:
	Picviz::PVAxis const* lib_axis() const;
	void set_axis_text_value(QGraphicsTextItem* item, PVRow const r);
	inline bool show_min_max_values() const { return _minmax_visible; }

	void update_axis_label_position();
	void update_axis_min_max_position();
	void update_layer_min_max_position();

private:
	PVSlidersManager_p              _sliders_manager_p;
	axis_id_t                       _axis_id;
	QRectF                          _bbox;
	Picviz::PVView const&           _lib_view;
	PVSlidersGroup                 *_sliders_group;
	PVAxisLabel                    *_label;
	PVAxisHeader              	   *_header_zone;
	int                             _axis_length;
	int                             _zone_width;
	QGraphicsTextItem              *_axis_min_value;
	QGraphicsTextItem              *_axis_max_value;
	QGraphicsTextItem              *_layer_min_value;
	QGraphicsTextItem              *_layer_max_value;
	__impl::PVToolTipEventFilter   *_event_filter;
	bool                            _minmax_visible;
};

}

#endif // PVPARALLELVIEW_PVAXISGRAPHICSITEM_H
