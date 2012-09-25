
#ifndef PVPARALLELVIEW_PVABSTRACTRANGEAXISSLIDERS_H
#define PVPARALLELVIEW_PVABSTRACTRANGEAXISSLIDERS_H

#include <pvparallelview/PVAbstractAxisSliders.h>
#include <pvparallelview/PVAbstractAxisSlider.h>

#include <QGraphicsItem>

namespace PVParallelView
{

class PVSlidersGroup;

class PVAbstractRangeAxisSliders : public PVAbstractAxisSliders
{
public:
	typedef PVSlidersManager::id_t id_t;
	typedef std::pair<PVRow, PVRow> range_t;

public:

	PVAbstractRangeAxisSliders(QGraphicsItem *parent, PVSlidersManager_p sm_p,
	                           PVSlidersGroup *group, const char *text);

	virtual void initialize(id_t id, uint32_t y_min, uint32_t y_max) = 0;

	virtual bool is_moving() const
	{
		return (_sl_min->is_moving() || _sl_max->is_moving());
	}

	range_t get_range() const
	{
		PVRow v_min = _sl_min->value();
		PVRow v_max = _sl_max->value();

		return std::make_pair(PVCore::min(v_min, v_max),
		                      PVCore::max(v_min, v_max));
	}

protected:
	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
	                   QWidget *widget = 0);

protected:
	PVAbstractAxisSlider *_sl_min;
	PVAbstractAxisSlider *_sl_max;
};

}

#endif // PVPARALLELVIEW_PVABSTRACTRANGEAXISSLIDERS_H
