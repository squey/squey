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

	/**MAX
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

	/**
	 * Update the gradient according to the scale and max count.
	 *
	 */
	void update_gradient();

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

/**
 * Short: This class is required to make sure the 2 spinboxes always have the
 * same geometry.
 *
 * Long: the Qt's spinbox geometry depends on its maximum and minimum values.
 *.But as the right spinbox's value is the left spinbox's maximum, the latter's
 * size may varying (when the value's order changes). Size policy and QLayout's
 * fonctionnalities do not permit to synchronize (or do as) spinboxes geometry.
 * This class allows a spinbox to synchronize its size with an other spinbox's
 * size (by having the biggest of the two).
 */
class PVMimeticDoubleSpinBox : public QDoubleSpinBox
{
public:
	PVMimeticDoubleSpinBox(QDoubleSpinBox* other = nullptr) :
	_other(other)
	{}

	void set_other(QDoubleSpinBox* other)
	{
		_other = other;
	}

	QSize sizeHint() const
	{
		QSize lsize = QDoubleSpinBox::sizeHint();

		if (_other == nullptr) {
			return lsize;
		}

		QSize fsize = _other->QDoubleSpinBox::sizeHint();

		return QSize(qMax(lsize.width(), fsize.width()),
		             qMax(lsize.height(), fsize.height()));
	}

	QSize minimumSizeHint() const
	{
		QSize lsize = QDoubleSpinBox::minimumSizeHint();

		if (_other == nullptr) {
			return lsize;
		}

		QSize fsize = _other->QDoubleSpinBox::minimumSizeHint();
		return QSize(qMax(lsize.width(), fsize.width()),
		             qMax(lsize.height(), fsize.height()));
	}

public:
	void use_floating_point(bool floating_point)
	{
		_use_floating_point = floating_point;
	}

protected:
	virtual QString textFromValue(double value) const override
	{
		// Using QLocale::toString(double) with high values returns QString as scientific notation,
		// (hence the cast to qulonglong).

		if (_use_floating_point) {
			return locale().toString(value, 'f', decimals());
		}
		else {
			return locale().toString((qulonglong)value);
		}
	}

private:
	QDoubleSpinBox *_other;
	bool _use_floating_point;
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
	PVAbstractRangePicker(
		const double& min_limit,
	    const double& max_limit,
	    QWidget* parent = nullptr
	);

	/**
	 * Set the range's minimum value
	 *
	 * @param value the new minimum value
	 */
	void set_range_min(const double& value, bool force = false);

	/**
	 * Get the range's minimum value
	 *
	 * @return the minimum value
	 */
	double get_range_min() const;

	/**
	 * Set the range's maximum value
	 *
	 * @param value the new maximum value
	 */
	void set_range_max(const double& value, bool force = false);

	/**
	 * Get the range's maximum value
	 *
	 * @return the maximum value
	 */
	double get_range_max() const;

	/**
	 * Convert a value to another representation.
	 * Base implementation does nothing, derived implementations can work with percentage for example.
	 *
	 * @return the converted value
	 */
	virtual double convert_to(const double& value) const { return value; }

	/**
	 * Convert a value from another representation.
	 * Base implementation does nothing, derived implementations can work with percentage for example.
	 *
	 * @return the converted value
	 */
	virtual double convert_from(const double& value) const { return value; }

public:
	void connect_ranges_to_spinboxes();
	void disconnect_ranges_from_spinboxes();

	void connect_spinboxes_to_ranges();
	void disconnect_spinboxes_from_ranges();

public:
	/**
	 * Set the lower and upper bound limits
	 *
	 * @param min_limit the lower bound
	 * @param max_limit the upper bound
	 */
	void set_limits(const double& min_limit,
	                const double& max_limit);

	/**
	 * get the lower bound limit
	 *
	 * @return the widget's lower bound limit
	 */
	double get_limit_min() const { return _limit_min; }

	/**
	 * get the uper bound limit
	 *
	 * @return the widget's uper bound limit
	 */
	double get_limit_max() const { return _limit_max; }

	/**
	 * get the limit's range
	 *
	 * @return the widget's limit's range
	 */
	double get_limit_range() const { return _limit_range; }

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
	 * @param linear the linear color gradient to use
	 * @param log the logarithmic color gradient to use
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

protected:
	/**
	 * Converts a value from spinbox's value space to color ramp's value space
	 *
	 * This method is helful to have non linear color ramp.
	 *
	 * @param value a value in the range accepted by spinboxes
	 *
	 * @return a value in the range [0;1]
	 */
	virtual double map_to_spinbox(const double& value) const;

	/**
	 * Converts a value from color ramp's value space to spinbox's value space
	 *
	 * This method is helful to have non linear color ramp.
	 *
	 * @param value a value in the range [0,1]
	 *
	 * @return a value in the range accepted by spinboxes
	 */
	virtual double map_from_spinbox(const double& value) const;

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

protected:
	__impl::PVAbstractRangeRamp*    _range_ramp;
	__impl::PVMimeticDoubleSpinBox* _min_spinbox;
	__impl::PVMimeticDoubleSpinBox* _max_spinbox;
	double                          _limit_min;
	double						    _min;
	double						    _max;
	double                          _limit_max;
	double                          _limit_range;
	double                          _epsilon;
};

}

#endif // PVWIDGETS_PVABSTRACTRANGEPICKER_H
