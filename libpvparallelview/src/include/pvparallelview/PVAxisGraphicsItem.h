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

class PVAxisLabel;

class PVAxisGraphicsItem : public QObject, public QGraphicsItemGroup
{
	Q_OBJECT

public:
	typedef PVSlidersGroup::selection_ranges_t selection_ranges_t;
	typedef PVSlidersManager::axis_id_t        axis_id_t;

	// Used to draw the axis out of the image zone
	constexpr static int axis_extend = 8;

public:
	PVAxisGraphicsItem(PVSlidersManager_p sm_p, Picviz::PVView const& view, const axis_id_t &axis_id);
	~PVAxisGraphicsItem();

	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

	void update_axis_info();

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

	QRectF get_label_scene_bbox() const;

	void set_axis_length(int l)
	{
		_axis_length = l;
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

public slots:
	void emit_new_zoomed_parallel_view(int axis_id)
	{
		emit new_zoomed_parallel_view(axis_id);
	}

signals:
	void new_zoomed_parallel_view(int axis_id);

private:
	Picviz::PVAxis const* lib_axis() const;

private:
	PVSlidersManager_p              _sliders_manager_p;
	axis_id_t                       _axis_id;
	QRectF                          _bbox;
	Picviz::PVView const&           _lib_view;
	PVSlidersGroup                 *_sliders_group;
	PVAxisLabel                    *_label;
	int                             _axis_length;
};

}

#endif // PVPARALLELVIEW_PVAXISGRAPHICSITEM_H