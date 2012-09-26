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

class PVAxisGraphicsItem : public QObject, public QGraphicsItemGroup
{
public:
	typedef PVSlidersGroup::selection_ranges_t selection_ranges_t;
	typedef PVSlidersManager::axe_id_t         axe_id_t;

public:
	PVAxisGraphicsItem(PVSlidersManager_p sm_p, Picviz::PVView const& view, const axe_id_t &axe_id);
	~PVAxisGraphicsItem();

	QRectF boundingRect () const;

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

	const axe_id_t get_axe_id() const
	{
		return _axe_id;
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

private:
	Picviz::PVAxis const* lib_axis() const;

private:
	PVSlidersManager_p              _sliders_manager_p;
	axe_id_t                        _axe_id;
	QRectF                          _bbox;
	Picviz::PVView const&           _lib_view;
	PVSlidersGroup                 *_sliders_group;
	QGraphicsSimpleTextItem        *_label;
};

}

#endif // PVPARALLELVIEW_PVAXISGRAPHICSITEM_H
