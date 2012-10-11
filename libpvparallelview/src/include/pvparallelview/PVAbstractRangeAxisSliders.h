
#ifndef PVPARALLELVIEW_PVABSTRACTRANGEAXISSLIDERS_H
#define PVPARALLELVIEW_PVABSTRACTRANGEAXISSLIDERS_H

#include <pvparallelview/PVAbstractAxisSliders.h>
#include <pvparallelview/PVAbstractAxisSlider.h>

class QGraphicsItem;

namespace PVParallelView
{

class PVSlidersGroup;

class PVAbstractRangeAxisSliders : public PVAbstractAxisSliders
{
public:
	typedef PVSlidersManager::id_t id_t;
	typedef std::pair<int64_t, int64_t> range_t;

public:

	PVAbstractRangeAxisSliders(QGraphicsItem *parent, PVSlidersManager_p sm_p,
	                           PVSlidersGroup *group, const char *text);

	virtual ~PVAbstractRangeAxisSliders();

	virtual void initialize(id_t id, int64_t y_min, int64_t y_max) = 0;

	virtual bool is_moving() const
	{
		return (_sl_min->is_moving() || _sl_max->is_moving());
	}

	range_t get_range() const
	{
		int64_t v_min = _sl_min->get_value();
		int64_t v_max = _sl_max->get_value();

		return std::make_pair(PVCore::min(v_min, v_max),
		                      PVCore::max(v_min, v_max));
	}

	virtual void refresh()
	{
		refresh_value(_sl_min->get_value(),
		              _sl_max->get_value());
	}

protected:
	void refresh_value(int64_t y_min, int64_t y_max)
	{
		_sl_min->set_value(y_min);
		_sl_max->set_value(y_max);
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
