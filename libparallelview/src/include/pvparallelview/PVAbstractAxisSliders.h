
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

	virtual bool is_moving() const = 0;

public slots:
	void remove_from_axis()
	{
		do_remove_from_axis();
	}

signals:
	void sliders_moved();

protected:
	virtual void do_remove_from_axis() = 0;

protected:
	PVSlidersManager_p       _sliders_manager_p;
	PVSlidersGroup          *_group;
	QGraphicsSimpleTextItem *_text;
};

}

#endif // PVPARALLELVIEW_PVABSTRACTAXISSLIDERS_H
