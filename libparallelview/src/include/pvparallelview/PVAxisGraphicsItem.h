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

// Used to draw the axis out of the image zone
#define PVAW_CST 8

namespace PVParallelView
{

typedef std::pair<PVAxisSlider*, PVAxisSlider*> PVAxisRangeSliders;

class PVAxisGraphicsItem : public QObject, public QGraphicsItemGroup
{
	Q_OBJECT

private:
	friend class zoom_sliders_new_obs;

public:
	typedef std::vector<std::pair<PVRow, PVRow> > selection_ranges_t;

public:
	PVAxisGraphicsItem(PVSlidersManager_p sm_p, Picviz::PVAxis *axis, uint32_t axis_index);

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
	class zoom_sliders_new_obs :
		public PVHive::PVFuncObserverSignal<PVSlidersManager,
		                                    FUNC(PVSlidersManager::new_zoom_sliders)>
	{
	public:
		zoom_sliders_new_obs(PVAxisGraphicsItem *parent) : _parent(parent)
		{}

		void update(arguments_deep_copy_type const& args) const
		{
			PVCol axis = std::get<0>(args);

			if (axis == _parent->_axis_index) {
				PVSlidersManager::id_t id = std::get<1>(args);
				uint32_t y_min = std::get<2>(args);
				uint32_t y_max = std::get<3>(args);
				printf("##### PVAxisGraphicsItem::zoom_sliders_new_obs: add new zoom sliders: %d %p %u %u\n",
				       axis, id, y_min, y_max);
				_parent->add_range_sliders(y_min, y_max);
			}
		}

	private:
		PVAxisGraphicsItem *_parent;
	};

private:
	PVSlidersManager_p              _sliders_manager_p;
	zoom_sliders_new_obs            _zsn_obs;
	Picviz::PVAxis*                 _axis;
	PVZoneID			_axis_index;
	QRectF                          _bbox;
	std::vector<PVAxisRangeSliders> _sliders;
};

}

#endif // PVPARALLELVIEW_PVAXISGRAPHICSITEM_H
