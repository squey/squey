//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/widgets/PVAbstractRangePicker.h>

#include <cmath> // for round

#include <QBrush>   // for QLinearGradient, QBrush
#include <QColor>   // for QColor
#include <QEvent>   // for QMouseEvent, QPaintEvent, etc
#include <QPainter> // for QPainter
#include <QPoint>   // for QPoint, QPointF
#include <QPolygon> // for QPolygon, QPolygonF
#include <QRect>    // for QRect
#include <QVBoxLayout>
#include <QWidget> // for QWidget
#include <QMouseEvent>

/**
 * TODO: to make thinks really really really cleaner/simpler:
 * - PVAbstractRangeRampCursor::paintEvent must use its geometry to draw the
 *   QPolygons, not fancy constants.
 * - PVAbstractRangeRampCursor must provide its own ::move() method to use
 *   the offset related to its type; a "[gs]etValue()" is a good start point
 * - PVAbstractRangeRamp::paintEvent must use its geometry to draw the
 *   gradients, not fancy constants;
 * - in PVAbstractRangeRamp, the ramp's geometry has to be stored in a QRect
 * - PVAbstractRangeRamp::resizeEvent has to be reimplemented to update the
 *   ramp geometry and the cursors geometry
   - surely lots of others step before removing *all* of the fancy constants
 *
 * One last point: using constants to do horology arithmetic is *always*
 * wrong! Why? Because it always hides an unclear thought...
 */
/**
 * some constants to parametrize the range picker
 */

// to define the margin between the ramp and the spinboxes
#define SPINBOX_TOP_MARGIN 4

// the color ramp's height
#define RAMP_HEIGHT 16

// the margin's size around the color ramp
#define RAMP_MARGIN 10

// the cursors side size
#define CURSOR_SIDE 8

// the additional horizontal distance used to grab the cursors
#define CURSOR_EXTRA_HGRAB 6

/**
 * some internal constants (do not edit until you know what you do:)
 */
#define CURSOR_VOFFSET (RAMP_MARGIN - (CURSOR_SIDE + 1))

#define CURSOR_WIDTH (CURSOR_SIDE + CURSOR_EXTRA_HGRAB)

#define CURSOR_HEIGHT (RAMP_HEIGHT + 2 * (CURSOR_SIDE + 1))

#define MINIMUM_CURSOR_OFFSET (RAMP_MARGIN - CURSOR_WIDTH)

#define RANGE_RAMP_HEIGHT (RAMP_HEIGHT + 2 * RAMP_MARGIN)

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRampCursor::PVAbstractRangeRampCursor
 *****************************************************************************/

PVWidgets::__impl::PVAbstractRangeRampCursor::PVAbstractRangeRampCursor(cursor_type type,
                                                                        QWidget* parent)
    : QWidget(parent), _type(type)
{
	setCursor(Qt::OpenHandCursor);

	setGeometry(0, 0, CURSOR_WIDTH, CURSOR_HEIGHT);
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRampCursor::paintEvent
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRampCursor::paintEvent(QPaintEvent* event)
{
	if (parentWidget() == nullptr) {
		return;
	}

	QPainter painter(this);

	QPolygonF triangle_up;
	QPolygon triangle_down;

	if (_type == MINIMUM) {
		/* Qt is a dunce to draw correctly this triangle; the .1 is
		 * required to have a nice isosceles rectangle triangle...
		 */
		triangle_up << QPointF(CURSOR_EXTRA_HGRAB - 1, 0)
		            << QPointF(CURSOR_EXTRA_HGRAB + CURSOR_SIDE - 1, 0)
		            << QPointF(CURSOR_EXTRA_HGRAB + CURSOR_SIDE - 1, CURSOR_SIDE + .1);
	} else {
		// for this one, there is no wrong drawing
		triangle_up << QPointF(0, 0) << QPointF(CURSOR_SIDE, 0) << QPointF(0, CURSOR_SIDE);
	}

	painter.setBrush(Qt::SolidPattern);
	painter.drawConvexPolygon(triangle_up);

	// no problem for the bottom triangles...
	if (_type == MINIMUM) {
		triangle_down << QPoint(CURSOR_EXTRA_HGRAB - 1, CURSOR_HEIGHT - 1)
		              << QPoint(CURSOR_EXTRA_HGRAB + CURSOR_SIDE - 1, CURSOR_HEIGHT - 1)
		              << QPoint(CURSOR_EXTRA_HGRAB + CURSOR_SIDE - 1,
		                        CURSOR_HEIGHT - CURSOR_SIDE - 1);
	} else {
		triangle_down << QPoint(0, CURSOR_HEIGHT - 1) << QPoint(CURSOR_SIDE, CURSOR_HEIGHT - 1)
		              << QPoint(0, CURSOR_HEIGHT - CURSOR_SIDE - 1);
	}

	painter.setBrush(QBrush(Qt::white));
	painter.drawConvexPolygon(triangle_down);
	event->accept();
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRampCursor::mousePressEvent
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRampCursor::mousePressEvent(QMouseEvent* event)
{
	QWidget::mousePressEvent(event);

	if (event->button() != Qt::LeftButton) {
		return;
	}

	setCursor(Qt::ClosedHandCursor);

	/* to avoid having a teleport effect when starting moving a cursor, it
	 * is a good idea to keep the offset between the mouse cursor and the moving ramp
	 * cursor
	 */
	if (_type == MINIMUM) {
		/* the cursor's geometry must be involved in the minimum cursor
		 * case... with f**ing constant to have correct value...
		 */
		_move_offset = event->pos().x() - CURSOR_SIDE + 2;
	} else {
		// the maximum cursor has no offset problem
		_move_offset = event->pos().x();
	}

	event->accept();
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRampCursor::mouseReleaseEvent
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRampCursor::mouseReleaseEvent(QMouseEvent* event)
{
	QWidget::mouseReleaseEvent(event);

	if (event->button() != Qt::LeftButton) {
		return;
	}

	setCursor(Qt::OpenHandCursor);
	event->accept();
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRampCursor::mouseMoveEvent
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRampCursor::mouseMoveEvent(QMouseEvent* event)
{
	QWidget::mouseMoveEvent(event);

	if (event->buttons() != Qt::LeftButton) {
		return;
	}

	Q_EMIT moved(mapToParent(event->pos()).x() - _move_offset);

	event->accept();
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRamp::PVAbstractRangeRamp
 *****************************************************************************/

PVWidgets::__impl::PVAbstractRangeRamp::PVAbstractRangeRamp(QWidget* parent) : QWidget(parent)
{
	setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
	setMinimumHeight(RANGE_RAMP_HEIGHT);
	setContentsMargins(0, 0, 0, 0);

	_min_cursor = new PVAbstractRangeRampCursor(PVAbstractRangeRampCursor::MINIMUM, this);
	connect(_min_cursor, &PVAbstractRangeRampCursor::moved, this,
	        &PVAbstractRangeRamp::min_cursor_moved);

	_max_cursor = new PVAbstractRangeRampCursor(PVAbstractRangeRampCursor::MAXIMUM, this);
	connect(_max_cursor, &PVAbstractRangeRampCursor::moved, this,
	        &PVAbstractRangeRamp::max_cursor_moved);
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRamp::set_gradient
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRamp::set_gradient(const QLinearGradient& gradient)
{
	_gradient = gradient;
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRamp::set_min_cursor
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRamp::set_min_cursor(double value)
{
	// range ramp's denormalization to pass it to the cursor
	_min_cursor->move(std::round(MINIMUM_CURSOR_OFFSET + (value * get_real_width())),
	                  CURSOR_VOFFSET);
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRamp::set_max_cursor
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRamp::set_max_cursor(double value)
{
	// range ramp's denormalization to pass it to the cursor
	_max_cursor->move(std::round(RAMP_MARGIN + (value * get_real_width())), CURSOR_VOFFSET);
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRamp::paintEvent
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRamp::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);

	/* first, the whole color ramp is drawn in "selected" mode
	 */
	QRect area(RAMP_MARGIN, RAMP_MARGIN, rect().width() - (2 * RAMP_MARGIN), RAMP_HEIGHT);

	painter.setClipping(false);
	painter.setPen(Qt::NoPen);

	_gradient.setStart(area.left(), 0);
	_gradient.setFinalStop(area.right(), 0);

	painter.setBrush(_gradient);

	painter.drawRect(area);

	painter.setBrush(QColor(0, 0, 0, 127));

	/* second, the "unselected" part are drawn using semi-transparent
	 * black rectangles.
	 */
	int end = _min_cursor->pos().x() + CURSOR_WIDTH;

	if (end >= area.left()) {
		QRect left_mask(RAMP_MARGIN, RAMP_MARGIN, end - area.left(), area.height());

		painter.drawRect(left_mask);
	}

	int start = _max_cursor->pos().x();

	if (start <= area.right()) {
		QRect right_mask(start, RAMP_MARGIN, area.right() - start + 1, area.height());
		painter.drawRect(right_mask);
	}
	event->accept();
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRamp::mousePressEvent
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRamp::mousePressEvent(QMouseEvent* event)
{
	fast_move(event);
	event->accept();
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRamp::mouseReleaseEvent
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRamp::mouseReleaseEvent(QMouseEvent* event)
{
	fast_move(event);
	event->accept();
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRamp::mouseMoveEvent
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRamp::mouseMoveEvent(QMouseEvent* event)
{
	fast_move(event);
	event->accept();
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRamp::fast_move
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRamp::fast_move(QMouseEvent* event)
{
	if (event->buttons() == Qt::LeftButton) {
		min_cursor_moved(event->pos().x() - CURSOR_SIDE - 1);
	} else if (event->buttons() == Qt::RightButton) {
		max_cursor_moved(event->pos().x());
	}
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRamp::get_real_width
 *****************************************************************************/

int PVWidgets::__impl::PVAbstractRangeRamp::get_real_width() const
{
	return rect().width() - 2 * RAMP_MARGIN;
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRamp::min_cursor_moved
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRamp::min_cursor_moved(int value)
{
	// cursor's value normalization to pass it to the spinbox
	Q_EMIT min_changed(value / (double)get_real_width());
}

/*****************************************************************************
 * PVWidgets::__impl::PVAbstractRangeRamp::max_cursor_moved
 *****************************************************************************/

void PVWidgets::__impl::PVAbstractRangeRamp::max_cursor_moved(int value)
{
	// cursor's value normalization to pass it to the spinbox
	Q_EMIT max_changed((value - RAMP_MARGIN) / (double)get_real_width());
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::PVAbstractRangePicker
 *****************************************************************************/

PVWidgets::PVAbstractRangePicker::PVAbstractRangePicker(const double& min_limit,
                                                        const double& max_limit,
                                                        QWidget* parent)
    : QWidget(parent)
    , _limit_min(min_limit)
    , _min(min_limit)
    , _max(max_limit)
    , _limit_max(max_limit)
    , _limit_range(max_limit - min_limit)
    , _epsilon(0.)
{
	setContentsMargins(2, 2, 2, 2);

	auto vl = new QVBoxLayout;
	vl->setContentsMargins(0, 0, 0, 0);
	vl->setSpacing(0);
	setLayout(vl);

	_range_ramp = new __impl::PVAbstractRangeRamp();
	vl->addWidget(_range_ramp);

	connect_ranges_to_spinboxes();

	auto hl = new QHBoxLayout;
	hl->setContentsMargins(RAMP_MARGIN, SPINBOX_TOP_MARGIN, RAMP_MARGIN, 0);
	hl->setSpacing(0);
	vl->addLayout(hl);
	vl->setAlignment(hl, Qt::AlignTop);

	_min_spinbox = new __impl::PVMimeticDoubleSpinBox;
	_min_spinbox->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	_min_spinbox->setAlignment(Qt::AlignRight);

	hl->addWidget(_min_spinbox, 0, Qt::AlignLeft);

	_max_spinbox = new __impl::PVMimeticDoubleSpinBox;
	_max_spinbox->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	_max_spinbox->setAlignment(Qt::AlignRight);

	hl->addWidget(_max_spinbox, 0, Qt::AlignRight);

	connect_spinboxes_to_ranges();

	static_cast<__impl::PVMimeticDoubleSpinBox*>(_min_spinbox)->set_other(_max_spinbox);
	static_cast<__impl::PVMimeticDoubleSpinBox*>(_max_spinbox)->set_other(_min_spinbox);

	/**
	 * the spinboxes are initialized/connected/whatever, now its time to update
	 * our little world :-)
	 */
	set_limits(min_limit, max_limit);
	set_range_min(min_limit);
	set_range_max(max_limit);
}

void PVWidgets::PVAbstractRangePicker::connect_ranges_to_spinboxes()
{
	connect(_range_ramp, &__impl::PVAbstractRangeRamp::min_changed, this,
	        &PVAbstractRangePicker::min_ramp_changed);
	connect(_range_ramp, &__impl::PVAbstractRangeRamp::max_changed, this,
	        &PVAbstractRangePicker::max_ramp_changed);
}

void PVWidgets::PVAbstractRangePicker::disconnect_ranges_from_spinboxes()
{
	disconnect(_range_ramp, &__impl::PVAbstractRangeRamp::min_changed, this,
	           &PVAbstractRangePicker::min_ramp_changed);
	disconnect(_range_ramp, &__impl::PVAbstractRangeRamp::max_changed, this,
	           &PVAbstractRangePicker::max_ramp_changed);
}

void PVWidgets::PVAbstractRangePicker::connect_spinboxes_to_ranges()
{
	connect(_min_spinbox, SIGNAL(valueChanged(double)), this, SLOT(min_spinbox_changed(double)));
	connect(_max_spinbox, SIGNAL(valueChanged(double)), this, SLOT(max_spinbox_changed(double)));
}

void PVWidgets::PVAbstractRangePicker::disconnect_spinboxes_from_ranges()
{
	disconnect(_min_spinbox, SIGNAL(valueChanged(double)), this, SLOT(min_spinbox_changed(double)));
	disconnect(_max_spinbox, SIGNAL(valueChanged(double)), this, SLOT(max_spinbox_changed(double)));
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::set_range_min
 *****************************************************************************/

void PVWidgets::PVAbstractRangePicker::set_range_min(const double& value, bool force /* = false */
                                                     )
{
	if (force) {
		_min_spinbox->blockSignals(true);
		_min_spinbox->setValue(_limit_min);
		_min_spinbox->blockSignals(false);
	}
	_min_spinbox->setValue(value);
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::get_range_min
 *****************************************************************************/

double PVWidgets::PVAbstractRangePicker::get_range_min() const
{
	return _min;
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::set_range_max
 *****************************************************************************/

void PVWidgets::PVAbstractRangePicker::set_range_max(const double& value, bool force /* = false */
                                                     )
{
	if (force) {
		_max_spinbox->blockSignals(true);
		_max_spinbox->setValue(_limit_max);
		_max_spinbox->blockSignals(false);
	}
	_max_spinbox->setValue(value);
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::get_range_max
 *****************************************************************************/

double PVWidgets::PVAbstractRangePicker::get_range_max() const
{
	return _max;
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::set_limits
 *****************************************************************************/

void PVWidgets::PVAbstractRangePicker::set_limits(const double& min_limit, const double& max_limit)
{
	_min_spinbox->setMinimum(min_limit);
	_min_spinbox->setMaximum(max_limit);

	_max_spinbox->setMinimum(min_limit);
	_max_spinbox->setMaximum(max_limit);
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::resizeEvent
 *****************************************************************************/

void PVWidgets::PVAbstractRangePicker::resizeEvent(QResizeEvent* event)
{
	QWidget::resizeEvent(event);

	/* when the widget is resized, the color ramp cursors have to
	 * update their position on the screen
	 */
	min_spinbox_changed(_min_spinbox->value());
	max_spinbox_changed(_max_spinbox->value());
	event->accept();
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::set_gradient
 *****************************************************************************/
void PVWidgets::PVAbstractRangePicker::set_gradient(const QLinearGradient& gradient)
{
	_range_ramp->set_gradient(gradient);
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::set_epsilon
 *****************************************************************************/

void PVWidgets::PVAbstractRangePicker::set_epsilon(const double& epsilon)
{
	_epsilon = epsilon;
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::map_from_spinbox
 *****************************************************************************/

double PVWidgets::PVAbstractRangePicker::map_from_spinbox(const double& value) const
{
	return (value - _limit_min) / _limit_range;
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::map_to_spinbox
 *****************************************************************************/

double PVWidgets::PVAbstractRangePicker::map_to_spinbox(const double& value) const
{
	return (value * _limit_range) + _limit_min;
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::min_spinbox_changed
 *****************************************************************************/

void PVWidgets::PVAbstractRangePicker::min_spinbox_changed(double value)
{
	_min = convert_from(value);

	// spinbox's value normalization to pass it to the range ramp
	_range_ramp->set_min_cursor(map_from_spinbox(value));

	_max_spinbox->setMinimum(value + _epsilon);
	update();
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::max_spinbox_changed
 *****************************************************************************/

void PVWidgets::PVAbstractRangePicker::max_spinbox_changed(double value)
{
	_max = convert_from(value);

	// spinbox's value normalization to pass it to the range ramp
	_range_ramp->set_max_cursor(map_from_spinbox(value));

	_min_spinbox->setMaximum(value - _epsilon);
	update();
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::min_ramp_changed
 *****************************************************************************/

void PVWidgets::PVAbstractRangePicker::min_ramp_changed(double value)
{
	// range ramp's value denormalization to pass it to the spinbox
	_min_spinbox->setValue(map_to_spinbox(value));
}

/*****************************************************************************
 * PVWidgets::PVAbstractRangePicker::max_ramp_changed
 *****************************************************************************/

void PVWidgets::PVAbstractRangePicker::max_ramp_changed(double value)
{
	// range ramp's value denormalization to pass it to the spinbox
	_max_spinbox->setValue(map_to_spinbox(value));
}
