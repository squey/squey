/**
 * \file PVSelectionSquareGraphicsItem.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVSELECTIONSQUAREGRAPHICSITEM_H_
#define PVSELECTIONSQUAREGRAPHICSITEM_H_

#include <limits>

#include <QGraphicsScene>
#include <QGraphicsRectItem>
#include <QTimer>
#include <QObject>

#include <pvkernel/core/PVLogger.h>

#define VOLATILE_SELECTION_TIMER_MSEC 300
#define VOLATILE_COLOR QColor(255, 127, 36)
#define COMMITED_COLOR QColor(255, 0, 0)
#define PEN_WIDTH 2

namespace PVParallelView
{

class PVFullParallelScene;

class PVSelectionSquareGraphicsItem : public QObject, public QGraphicsRectItem
{
	Q_OBJECT;

public:
	PVSelectionSquareGraphicsItem(QGraphicsScene* s);

	void clear_rect()
	{
		setRect(QRect());
	}

	void update_rect(const QRectF & rectangle)
	{
		setRect(rectangle);
		handle_volatile_selection();
	}

	void finished()
	{
		QPen cur_pen = pen();
		cur_pen.setColor(COMMITED_COLOR);
		setPen(cur_pen);
		_volatile_selection_timer->stop();
	}

signals:
	void commit_volatile_selection();

private slots:
	void volatile_selection_timeout_Slot()
	{
		finished();
		emit commit_volatile_selection();
	}

private:
	void handle_volatile_selection()
	{
		// Change volatile selection color
		QPen cur_pen = pen();
		cur_pen.setColor(VOLATILE_COLOR);
		setPen(cur_pen);

		// Reset volatile selection timer interval
		_volatile_selection_timer->start(VOLATILE_SELECTION_TIMER_MSEC);
	}

private:
	QTimer* _volatile_selection_timer;
};

}

#endif /* PVSELECTIONSQUAREGRAPHICSITEM_H_ */
