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

	void set_zone_width(int w)
	{
		_zone_width = w;
	}

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
	int                             _axis_length;
	int                             _zone_width;
	QGraphicsTextItem              *_axis_min_value;
	QGraphicsTextItem              *_axis_max_value;
	QGraphicsTextItem              *_layer_min_value;
	QGraphicsTextItem              *_layer_max_value;
	__impl::PVToolTipEventFilter   *_event_filter;
	bool                            _minmax_visible;

	__impl::PVAxisSelectedAnimation   *_axis_selected_animation;
};

namespace __impl
{

class PVAxisSelectedAnimation : QObject
{
	Q_OBJECT

	Q_PROPERTY(qreal opacity READ get_opacity WRITE set_opacity);
	Q_PROPERTY(qreal blur READ get_blur WRITE set_blur);

private:
	static constexpr size_t opacity_animation_duration_ms = 100;
	static constexpr QEasingCurve::Type opacity_animation_easing = QEasingCurve::Linear;

	static constexpr qreal blur_animation_min_amount = 0.0;
	static constexpr qreal blur_animation_max_amount = 5.0;
	static constexpr size_t blur_animation_duration_ms = 800;
	static constexpr QEasingCurve::Type blur_animation_easing = QEasingCurve::InOutQuad;

public:
	PVAxisSelectedAnimation(PVAxisGraphicsItem* parent);
	~PVAxisSelectedAnimation()
	{
		delete _opacity_animation;
		delete _blur_animation;
	}

public:
	void start(bool start);

private: // properties
	qreal get_opacity() const { return 0.0; } // avoid Qt warning
	void set_opacity(qreal opacity);

	qreal get_blur() const { return 0.0; } // avoid Qt warning
	void set_blur(qreal blur);

private slots: // animation
	void blur_animation_current_loop_changed();

private:
	inline PVAxisGraphicsItem* axis() { return (PVAxisGraphicsItem*) parent(); }

private:
	QGraphicsPixmapItem* _selected_axis_hole;
	QGraphicsPixmapItem* _selected_axis_dot;

	QPropertyAnimation* _opacity_animation;
	QPropertyAnimation* _blur_animation;
};

}

}

#endif // PVPARALLELVIEW_PVAXISGRAPHICSITEM_H
