/**
 * \file PVSelectionRectangle.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef PVWIDGETS_PVSELECTIONRECTANGLE_H
#define PVWIDGETS_PVSELECTIONRECTANGLE_H

#include <QObject>
#include <QCursor>
#include <QPen>
#include <QTimer>
#include <QColor>

#include <vector>

#include <pvparallelview/PVSelectionRectangleItem.h>

class QGraphicsScene;

namespace Picviz
{

class PVView;

}

/**
 * @todo: les move_by de 1 pixel doivent tenir compte des transformations (bogue
 * déjà présent dans le carré de sélection)
 */

namespace PVParallelView
{

/**
 * @class PVSelectionRectangle
 *
 * a wrapper and an helper for PVSelectionRectangleItem
 */
class PVSelectionRectangle : public QObject
{
	Q_OBJECT;

public:
	enum SelectionMode
	{
		RECTANGLE,
		HORIZONTAL,
		VERTICAL
	};

public:
	static constexpr qreal GROW_STEP_RATIO = 1.2;

	static const QColor rect_color;
	static const QColor handle_color;

public:
	PVSelectionRectangle(QGraphicsScene* scene);
	virtual ~PVSelectionRectangle() {};

public:
	/**
	 * clear and hide the selection rectangle
	 */
	void clear();

	/**
	 * show and start a mouse interaction
	 */
	void begin(const QPointF& p);

	/**
	 * process a mouse interaction step
	 */
	void step(const QPointF& p);

	/**
	 * terminate a mouse interaction
	 */
	void end(const QPointF& p);

public:
	/**
	 */
	void move_left_by_step()
	{
		move_by(-_rect->get_handles_x_scale(), 0);
	}
	void move_right_by_step()
	{
		move_by(_rect->get_handles_x_scale(), 0);
	}
	void move_horizontally_by_step(bool left)
	{
		move_by((left ? -1 : 1 ) * _rect->get_handles_x_scale(), 0);
	}

	void move_up_by_step()
	{
		move_by(0, -_rect->get_handles_y_scale());
	}
	void move_down_by_step()
	{
		move_by(0, _rect->get_handles_y_scale());
	}
	void move_vertically_by_step(bool up)
	{
		move_by(0, (up ? -1 : 1) * _rect->get_handles_y_scale());
	}

	void move_left_by_width()
	{
		move_by(-get_rect().width(), 0);
	}
	void move_right_by_width()
	{
		move_by(get_rect().width(), 0);
	}
	void move_horizontally_by_width(bool left)
	{
		move_by((left ? -1 : 1) * get_rect().width(), 0);
	}

	void move_up_by_height()
	{
		move_by(0, -get_rect().height());
	}
	void move_down_by_height()
	{
		move_by(0, get_rect().height());
	}
	void move_vertically_by_height(bool up)
	{
		move_by(0, (up ? -1 : 1 ) * get_rect().height());
	}

	void grow_horizontally()
	{
		grow_by(GROW_STEP_RATIO, 1);
	}
	void shrink_horizontally()
	{
		grow_by(1/GROW_STEP_RATIO, 1);
	}
	void grow_vertically()
	{
		grow_by(1, 1/GROW_STEP_RATIO);
	}
	void shrink_vertically()
	{
		grow_by(1, GROW_STEP_RATIO);
	}

public:
	/**
	 * set the pen color to use to draw the rectangle
	 *
	 * @param color the color to use
	 */
	void set_pen_color(const QColor& col)
	{
		_rect->set_pen_color(col);
	}

	/**
	 * set the default mouse cursor to use
	 *
	 * @param cursor the default mouse cursor
	 */
	void set_default_cursor(QCursor cursor)
	{
		_rect->set_default_cursor(cursor);
	}

	/**
	 * get the default mouse cursor to use
	 *
	 * @return cursor the default mouse cursor
	 */
	QCursor get_default_cursor() const
	{
		return _rect->get_default_cursor();
	}

	/**
	 * set the pen color to use to draw the handles outline
	 *
	 * @param color the color to use
	 */
	void set_handles_pen_color(const QColor& col) const
	{
		_rect->set_handles_pen_color(col);
	}

	/**
	 * set the pen color to use to fill the handles
	 *
	 * using a color with an alpha != 0
	 *
	 * @param color the color to use
	 */
	void set_handles_brush_color(const QColor& col) const
	{
		_rect->set_handles_brush_color(col);
	}

	/**
	 * set the scale factors for handles
	 *
	 * @sa PVSelectionHandle::set_scale(const qreal xscale, const qreal yscale)
	 *
	 * @param xscale the scale factor along X
	 * @param yscale the scale factor along Y
	 */
	void set_handles_scale(const qreal xscale, const qreal yscale) const
	{
		_rect->set_handles_scale(xscale, yscale);
	}

	/**
	 * set the rectangle's geometry
	 *
	 * @param rect the rectangle
	 */
	void set_rect(const QRectF& rect)
	{
		return _rect->set_rect(rect);
	}

	/**
	 * get the selection rectangle's geometry
	 *
	 * @return the reactangle's geometry
	 */
	QRectF get_rect()
	{
		return _rect->get_rect();
	}

	/**
	 * get the selection rectangle's geometry (const version)
	 *
	 * @return the reactangle's geometry
	 */
	const QRectF get_rect() const
	{
		return _rect->get_rect();
	}

public:
	SelectionMode selection_mode() const
	{
		return _sel_mode;
	}

public slots:
	void set_selection_mode(int sel_mode)
	{
		_sel_mode = (SelectionMode)sel_mode;
	}

public:
	QGraphicsScene* scene() const;

signals:
	void commit_volatile_selection(bool use_selection_modifiers);

protected slots:
	void start_timer();

protected slots:
	virtual void commit(bool use_selection_modifiers) = 0;
	virtual Picviz::PVView& lib_view() = 0;

protected slots:
	void timeout();

private:
	void move_by(qreal hstep, qreal vstep);
	void grow_by(qreal hratio, qreal vratio);

private:
	PVSelectionRectangleItem* _rect;
	QTimer*                   _timer;
	bool                      _use_selection_modifiers;
	SelectionMode             _sel_mode;
};

}

#endif // PVWIDGETS_PVSELECTIONRECTANGLE_H
