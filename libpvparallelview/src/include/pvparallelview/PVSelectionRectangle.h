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
class QActionGroup;
class QToolBar;
class QSignalMapper;
class QToolButton;

namespace Picviz
{

class PVView;

}

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
	using SelectionMode = PVSelectionRectangleItem::SelectionMode;

public:
	static constexpr qreal GROW_STEP_RATIO = 1.2;

	static const QColor rectangle_color;
	static const QColor handle_color;
	static const int handle_transparency;
	static const int delay_msec;

public:
	PVSelectionRectangle(QGraphicsScene* scene);
	virtual ~PVSelectionRectangle() {};

public:
	/**
	 * clear and hide the selection rectangle
	 */
	virtual void clear();

	/**
	 * show and start a mouse interaction
	 */
	void begin(const QPointF& p);

	/**
	 * process a mouse interaction step
	 */
	void step(const QPointF& p);

	/**
	 * terminate a mouse interaction (with extra parameter:)
	 */
	void end(const QPointF& p, bool use_sel_modifiers = true, bool now = false);

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

public:
	/**
	 * set the Z value of the underlying QGraphicsItem
	 *
	 * @param zvalue the wanted Z value
	 */
	void set_z_value(qreal zvalue)
	{
		_rect->setZValue(zvalue);
	}

	/**
	 * do an update of the underlying QGraphicsItem
	 */
	void update()
	{
		_rect->update();
	}

	/**
	 * set the rectangle's geometry
	 *
	 * @param rect the rectangle
	 */
	void set_rect(const QRectF& rect, bool commit = true)
	{
		return _rect->set_rect(rect, commit);
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
	/**
	 * set the horizontal range for the rectangle
	 *
	 * @param min_value the lower bound value
	 * @param max_value the upper bound value
	 */
	void set_x_range(qreal min_value, qreal max_value)
	{
		_rect->set_x_range(min_value, max_value);
	}

	/**
	 * set the vertical range for the rectangle
	 *
	 * @param min_value the lower bound value
	 * @param max_value the upper bound value
	 */
	void set_y_range(qreal min_value, qreal max_value)
	{
		_rect->set_y_range(min_value, max_value);
	}


	/**
	 * clear the horizontal range for the rectangle
	 */
	void clear_x_range()
	{
		_rect->clear_x_range();
	}

	/**
	 * clear the vertical range for the rectangle
	 */
	void clear_y_range()
	{
		_rect->clear_y_range();
	}

public:
	/**
	 * get the current selection mode.
	 *
	 * @retun the current selection mode.
	 */
	SelectionMode selection_mode() const
	{
		return _rect->selection_mode();
	}

public slots:
	/**
	 * set the selection mode to use.
	 *
	 * @param sel_mode the new selection mode.
	 */
	void set_selection_mode(int sel_mode)
	{
		_rect->set_selection_mode(sel_mode);
	}

public:
	/**
	 * get the scene owning the selection rectangle
	 *
	 * @ return the scene owning the selection rectangle
	 */
	QGraphicsScene* scene() const;

public:
	/**
	 * create and add a selection mode selector
	 *
	 * @param view the parent view
	 * @param toolbar the toolbar containing the selector
	 * @param the signal mapper to use
	 *
	 * @return the added tool button
	 */
	static QToolButton* add_selection_mode_selector(QWidget *view,
	                                                QToolBar *toolbar,
	                                                QSignalMapper *signal_mapper);

	/**
	 * update a selection mode selector according to a given mode
	 *
	 * @param button the selector to update
	 * @param mode the new selection mode
	 */
	static void update_selection_mode_selector(QToolButton* button,
	                                           int mode);
signals:
	/**
	 * the signal which is fired when the volatile selection must
	 * be committed to the final selection
	 */
	void commit_volatile_selection(bool use_selection_modifiers);

protected slots:
	/**
	 * start the timer used to commit for selection.
	 */
	void start_timer();
	/**
	 * action done when the timer's timeout occurs.
	 */
	void timeout();

protected slots:
	/**
	 * method to override to implement selection commits
	 */
	virtual void commit(bool use_selection_modifiers) = 0;

	/**
	 * method to override to implement PVView retrieval
	 */
	virtual Picviz::PVView& lib_view() = 0;

private:
	void move_by(qreal hstep, qreal vstep);
	void grow_by(qreal hratio, qreal vratio);

private:
	PVSelectionRectangleItem* _rect;
	QTimer*                   _timer;
	bool                      _use_selection_modifiers;
};

}

#endif // PVWIDGETS_PVSELECTIONRECTANGLE_H
