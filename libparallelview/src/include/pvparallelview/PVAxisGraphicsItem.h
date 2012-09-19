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
#include <pvparallelview/PVAxisSlider.h>
#include <pvparallelview/PVSlidersManager.h>
#include <pvparallelview/PVSlidersGroup.h>

namespace PVParallelView
{

class PVAxisGraphicsItem : public QObject, public QGraphicsItemGroup
{
	friend class zoom_sliders_new_obs;

public:
	typedef PVSlidersGroup::selection_ranges_t selection_ranges_t;

public:
	PVAxisGraphicsItem(PVSlidersManager_p sm_p, Picviz::PVView const& view, uint32_t axis_index);
	~PVAxisGraphicsItem();

	QRectF boundingRect () const;

	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

	PVSlidersGroup *get_sliders_group()
	{
		return _sliders_group;
	}

	const PVSlidersGroup *get_sliders_group() const
	{
		return _sliders_group;
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
	PVZoneID			            _axis_index;
	QRectF                          _bbox;
	Picviz::PVView const&           _lib_view;
	PVSlidersGroup                 *_sliders_group;
};

}

#endif // PVPARALLELVIEW_PVAXISGRAPHICSITEM_H
