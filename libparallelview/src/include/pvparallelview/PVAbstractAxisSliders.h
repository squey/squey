
#ifndef PVPARALLELVIEW_PVABSTRACTAXISSLIDERS_H
#define PVPARALLELVIEW_PVABSTRACTAXISSLIDERS_H

#include <pvparallelview/PVSlidersManager.h>

#include <QObject>
#include <QGraphicsItemGroup>
#include <QGraphicsSimpleTextItem>

namespace PVParallelView
{

class PVSlidersGroup;

class PVAbstractAxisSliders : public QObject, public QGraphicsItemGroup
{
Q_OBJECT

public:
	PVAbstractAxisSliders(QGraphicsItem *parent, PVSlidersManager_p sm_p,
	                      PVSlidersGroup *group, const char *text);

	inline PVSlidersGroup *get_sliders_group() const
	{
		return _group;
	}

	virtual bool is_moving() const = 0;

	virtual void refresh() = 0;

public slots:
	virtual void remove_from_axis() = 0;

signals:
	void sliders_moved();

protected:
	PVSlidersManager_p       _sliders_manager_p;
	PVSlidersGroup          *_group;
	QGraphicsSimpleTextItem *_text;
};

}

#endif // PVPARALLELVIEW_PVABSTRACTAXISSLIDERS_H
