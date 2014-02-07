/**
 * \file PVAbstractRangePicker.h
 *
 * Copyright (C) Picviz Labs 2010-2014
 */

#ifndef PVWIDGETS_PVABSTRACTRANGEPICKER_H
#define PVWIDGETS_PVABSTRACTRANGEPICKER_H

#include <QWidget>
#include <QDoubleSpinBox>

class QLinearGradient;

namespace PVWidgets
{

namespace __impl
{

/**
 * @class PVAbstractRangeRampCursor
 *
 * This widget represents a cursor used in the PVAbstractRangePicker widget.
 *
 * Its logic depends on the fact it is the minimum cursor or the maximum one.
 * The value represented by the cursor is expressed in pixel along the color
 * ramp.
 */
class PVAbstractRangeRampCursor : public QWidget
{
	Q_OBJECT

public:
	enum cursor_type {
		MINIMUM  = 0,
		MAXIMUM = 1
	};

	/**
	 * CTOR
	 *
	 * @param type the cursor type (MINIMUM or MAXIMUM)
	 * @param parent the parent widget
	 */
	PVAbstractRangeRampCursor(cursor_type type, QWidget* parent = nullptr);

signals:
	/**
	 * this signal is emitted each time the cursor is moved using the mouse
	 */
	void moved(int value);

protected:
	/**
	 * Redefinition of QWidget::paintEvent
	 *
	 * @param event the involved paint event
	 */
	void paintEvent(QPaintEvent* event) override;

	/**
	 * Redefinition of QWidget::mousePressEvent
	 *
	 * @param event the involved mouse event
	 */
	void mousePressEvent(QMouseEvent* event) override;

	/**
	 * Redefinition of QWidget::mouseReleaseEvent
	 *
	 * @param event the involved mouse event
	 */
	void mouseReleaseEvent(QMouseEvent* event) override;

	/**
	 * Redefinition of QWidget::mouseMoveEvent
	 *
	 * @param event the involved mouse event
	 */
	void mouseMoveEvent(QMouseEvent* event) override;

private:
	cursor_type _type;
	int         _move_offset;
};

/**
 * @class PVAbstractRangeRamp
 *
 * This class represents the color ramp with its two cursors.
 *
 * To decorrelate the values of the spinboxes and of the cursors, this
 * widget uses normalized values in the range [0.0;1.0] (normalized values
 * for short).
 */
class PVAbstractRangeRamp : public QWidget
{
	Q_OBJECT

public:
	/**
	 * CTOR
	 *
	 * @param parent the parent widget
	 */
	PVAbstractRangeRamp(QWidget* parent = nullptr);

	/**
	 * Set the color gradient used for the color ramp
	 *
	 * @param gradient the color gradient to use
	 */
	void set_gradient(const QLinearGradient& gradient);

	/**
	 * Set the minimum value
	 *
	 * @param value thr gradient the used color gradient
	 */
	void set_min_cursor(double value);
	void set_max_cursor(double value);

protected:
	/**
	 * Redefinition of QWidget::paintEvent
	 *
	 * @param event the involved paint event
	 */
	void paintEvent(QPaintEvent* event) override;

	/**
	 * Redefinition of QWidget::mousePressEvent
	 *
	 * @param event the involved mouse event
	 */
	void mousePressEvent(QMouseEvent* event) override;

	/**
	 * Redefinition of QWidget::mouseReleaseEvent
	 *
	 * @param event the involved mouse event
	 */
	void mouseReleaseEvent(QMouseEvent* event) override;

	/**
	 * Redefinition of QWidget::mouseMoveEvent
	 *
	 * @param event the involved mouse event
	 */
	void mouseMoveEvent(QMouseEvent* event) override;

private:
	/**
	 * Process its parameter to move the cursor designated by
	 * the pressed mouse button
	 *
	 * @param event the involved mouse event
	 */
	void fast_move(QMouseEvent* event);

	/**
	 * Returns the width of the color ramp which has to be
	 * distinguished from the width of the whole widget
	 *
	 * @return the color ramp width
	 */
	int get_real_width() const;

private slots:
	/**
	 * This slot is called each time the minimum cursor is moved
	 *
	 * @param value the minimum cursor position
	 */
	void min_cursor_moved(int value);

	/**
	 * This slot is called each time the minimum cursor is moved
	 *
	 * @param value the minimum cursor position
	 */
	void max_cursor_moved(int value);

signals:
	/**
	 * This signal is emitted each time the minimum value is changed
	 *
	 * @param value the normalized minimum value
	 */
	void min_changed(double value);

	/**
	 * This signal is emitted each time the maximum value is changed
	 *
	 * @param value the normalized maximum value
	 */
	void max_changed(double value);

private:
	QLinearGradient            _gradient;
	PVAbstractRangeRampCursor* _min_cursor;
	PVAbstractRangeRampCursor* _max_cursor;
};

}

/**
 * @class PVAbstractRangePicker
 *
 * This widget helps choosing a range using 2 spinboxes or a color ramp
 * with 2 cursors.
 */
class PVAbstractRangePicker : public QWidget
{
	Q_OBJECT

public:
	/**
	 * CTOR
	 *
	 * @param min_limit the range lower value
	 * @param max_limit the range upper value
	 * @param widget the parent widget
	 */
	PVAbstractRangePicker(double min_limit, double max_limit, QWidget* parent = nullptr);

	/**
	 * set the minimum value
	 *
	 * @param value the new minimum value
	 */
	void set_min(const double& value);

	/**
	 * get the minimum value
	 *
	 * @return the minimum value
	 */
	double get_min() const;

	/**
	 * set the maximum value
	 *
	 * @param value the new maximum value
	 */
	void set_max(const double& value);

	/**
	 * get the maximum value
	 *
	 * @return the maximum value
	 */
	double get_max() const;

protected:
	/**
	 * Redefinition of QWidget::resizeEvent
	 *
	 * @param event the involved resize event
	 */
	void resizeEvent(QResizeEvent* event) override;

protected:
	/**
	 * Set the gradient used for the color ramp
	 *
	 * @param gradient the color gradient to use
	 *
	 * @note Use it to customize this spinbox in your derivated classe
	 */
	void set_gradient(const QLinearGradient& gradient);

	/**
	 * Set an epsilon value to separate the minimum value from the
	 * the maximum value
	 *
	 * @param epsilon the epsilon value
	 */
	void set_epsilon(const double& epsilon);

protected:
	/**
	 * Returns the spinbox used for the minimum value
	 *
	 * @note Use it to customize this spinbox in your derivated classe
	 */
	QDoubleSpinBox* get_min_spinbox() { return _min_spinbox; }

	/**
	 * Returns the spinbox used for the maximum value
	 *
	 * @note Use it to customize this spinbox in your derivated classe
	 */
	QDoubleSpinBox* get_max_spinbox() { return _max_spinbox; }

private slots:
	/**
	 * This slot is called each time the minimum spinbox's value has changed
	 *
	 * @param value the new minimum spinbox value
	 */
	void min_spinbox_changed(double value);
	/**
	 * This slot is called each time the maximum spinbox's value has changed
	 *
	 * @param value the new maximum spinbox value
	 */
	void max_spinbox_changed(double value);

	/**
	 * This slot is called each time the color ramp's minimum value has changed
	 *
	 * @param value the new color ramps minimum value
	 */
	void min_ramp_changed(double value);

	/**
	 * This slot is called each time the color ramp's maximum value has changed
	 *
	 * @param value the new color ramps maximum value
	 */
	void max_ramp_changed(double value);

private:
	__impl::PVAbstractRangeRamp* _range_ramp;
	QDoubleSpinBox*              _min_spinbox;
	QDoubleSpinBox*              _max_spinbox;
	double                       _limit_min;
	double                       _limit_max;
	double                       _limit_range;
	double                       _epsilon;
};

}

#endif // PVWIDGETS_PVABSTRACTRANGEPICKER_H
