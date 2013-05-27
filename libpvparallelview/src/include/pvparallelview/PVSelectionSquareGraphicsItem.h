/**
 * \file PVSelectionSquareGraphicsItem.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVSELECTIONSQUAREGRAPHICSITEM_H_
#define PVSELECTIONSQUAREGRAPHICSITEM_H_

#include <limits>

#include <QApplication>
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
	static constexpr unsigned int hsel_modifier = Qt::ControlModifier;
	static constexpr unsigned int vsel_modifier = Qt::ShiftModifier;

public:
	PVSelectionSquareGraphicsItem(QGraphicsItem* parent = nullptr);

	bool is_null() const
	{
		return rect().isNull();
	}

	void clear_rect()
	{
		setRect(QRect());
	}

	qreal top() const
	{
		return rect().top();
	}

	qreal bottom() const
	{
		return rect().bottom();
	}

	void update_rect(const QRectF & rectangle, bool use_selection_modifiers = true, bool now = false)
	{
		_use_selection_modifiers = use_selection_modifiers;
		update_rect_no_commit(rectangle);
		if (now) {
			volatile_selection_timeout_Slot();
		}
		else {
			handle_volatile_selection();
		}
	}

	void update_rect_no_commit(const QRectF & rectangle)
	{
		QRectF r = rectangle;

		if (_use_selection_modifiers) {
			unsigned int modifiers = (unsigned int) QApplication::keyboardModifiers() & ~Qt::KeypadModifier;
			if (vertical_selection_modifier() && modifiers == vertical_selection_modifier()) {
				r.setX(0);
				r.setWidth(scene()->sceneRect().width());
			}
			else if (horizontal_selection_modifier() && modifiers == horizontal_selection_modifier())
			{
				r.setY(0);
				r.setHeight(scene()->sceneRect().height());
			}
		}

		setRect(r);
	}

	void finished()
	{
		QPen cur_pen = pen();
		cur_pen.setColor(COMMITED_COLOR);
		setPen(cur_pen);
		_volatile_selection_timer->stop();
	}

	void enable_horizontal_selection(bool enabled)
	{
		_horizontal_selection_enabled = enabled;
	}

	unsigned int horizontal_selection_modifier()
	{
		if (_horizontal_selection_enabled) {
			return hsel_modifier;
		}
		return 0;
	}

	void enable_vertical_selection(bool enabled)
	{
		_vertical_selection_enabled = enabled;
	}

	unsigned int vertical_selection_modifier()
	{
		if (_vertical_selection_enabled) {
			return vsel_modifier;
		}
		return 0;
	}

protected:
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

signals:
	void commit_volatile_selection(bool use_selection_modifiers);

private slots:
	void volatile_selection_timeout_Slot()
	{
		finished();
		emit commit_volatile_selection(_use_selection_modifiers);
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
	QPointF _selection_square_pos;
	bool _use_selection_modifiers = true;
	bool _horizontal_selection_enabled = false;
	bool _vertical_selection_enabled = false;
};

}

#endif /* PVSELECTIONSQUAREGRAPHICSITEM_H_ */
