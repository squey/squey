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
#include <pvparallelview/common.h>
#include <pvparallelview/PVAxisSlider.h>
#include <picviz/PVAxis.h>

// Used to draw the axis out of the image zone
#define PVAW_CST 8

namespace PVParallelView
{

typedef std::pair<PVAxisSlider*, PVAxisSlider*> PVAxisRangeSliders;

class PVAxisGraphicsItem : public QObject, public QGraphicsItemGroup
{
	Q_OBJECT

public:
	typedef std::vector<std::pair<PVRow, PVRow> > selection_ranges_t;

public:
	PVAxisGraphicsItem(Picviz::PVAxis *axis, uint32_t axis_index);

	QRectF boundingRect () const;

	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

	void add_range_sliders(uint32_t y1, uint32_t y2);
	bool sliders_moving() const;

	QRect map_from_scene(QRectF rect) const
	{
		QPointF point = mapFromScene(rect.topLeft());
		return QRect(point.x(), point.y(), rect.width(), rect.height());
	}

	selection_ranges_t get_selection_ranges()
	{
		selection_ranges_t ranges;

		for (PVParallelView::PVAxisRangeSliders sliders : _sliders) {
			PVRow min = PVCore::min(sliders.first->value(), sliders.second->value());
			PVRow max = PVCore::max(sliders.first->value(), sliders.second->value());
			ranges.push_back(std::make_pair(min, max));
		}

		return ranges;
	}

signals:
	void axis_sliders_moved(PVZoneID);

protected slots:
	void slider_moved() { emit axis_sliders_moved(_axis_index); }

private:
	Picviz::PVAxis*                 _axis;
	PVZoneID						_axis_index;
	QRectF                          _bbox;
	std::vector<PVAxisRangeSliders> _sliders;
};

}

#endif // PVPARALLELVIEW_PVAXISGRAPHICSITEM_H
