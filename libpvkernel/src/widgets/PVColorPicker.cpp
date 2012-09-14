#include <pvkernel/widgets/PVColorPicker.h>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>

#define CROSS_HEIGHT 20
#define CROSS_WIDTH 20
#define CROSS_THICK 3

PVWidgets::PVColorPicker::PVColorPicker(QWidget* parent):
	QFrame(parent)
{
	init();
}

PVWidgets::PVColorPicker::PVColorPicker(PVCore::PVHSVColor const& c, QWidget* parent):
	QFrame(parent)
{
	init();
	set_color(c);
}

void PVWidgets::PVColorPicker::init()
{
	_offset = 0;
	setFocusPolicy(Qt::StrongFocus);
	setFrameShape(QFrame::Box);
	setFrameShadow(QFrame::Sunken);
	setLineWidth(1);
}

void PVWidgets::PVColorPicker::set_color(PVCore::PVHSVColor const& c)
{
	_c = c;
	_cross.rx() = h_to_x_screen(c.h());
	_cross.ry() = size().height()/2;
	_cross_y_rel = 0.5;
	update();
}

uint8_t PVWidgets::PVColorPicker::screen_x_to_h(const int x) const
{
	constexpr static size_t color_max = ((1<<HSV_COLOR_NBITS_ZONE)*6);
	const int width  = size().width();
	uint8_t h = (uint8_t) ((x*color_max)/width);
	h = (h+h_offset())%color_max;
	return h;
}

int PVWidgets::PVColorPicker::h_to_x_screen(const uint8_t h) const
{
	constexpr static size_t color_max = ((1<<HSV_COLOR_NBITS_ZONE)*6);
	const int width  = size().width();
	const int view_h = (h-h_offset())%color_max;
	return (view_h*width)/color_max;
}

QSize PVWidgets::PVColorPicker::sizeHint() const
{
	constexpr static size_t color_max = ((1<<HSV_COLOR_NBITS_ZONE)*6);
	return QSize(color_max, 10);
}

void PVWidgets::PVColorPicker::mousePressEvent(QMouseEvent* event)
{
	process_mouse_event(event);
}

void PVWidgets::PVColorPicker::mouseMoveEvent(QMouseEvent* event)
{
	process_mouse_event(event);
}

void PVWidgets::PVColorPicker::resizeEvent(QResizeEvent* /*event*/)
{
	_cross.rx() = h_to_x_screen(_c.h());
	_cross.ry() = (int) (_cross_y_rel*(double)size().height());
}

void PVWidgets::PVColorPicker::process_mouse_event(QMouseEvent* event)
{
	if ((event->buttons() & Qt::LeftButton) == Qt::LeftButton) {
		update_h(screen_x_to_h(event->x()));
		set_cross_pos(event->pos());
	}
}

void PVWidgets::PVColorPicker::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);
	QRect const& rect = event->rect();
	const int height = size().height();

	QRect draw_rect = rect.intersect(contentsRect());

	QColor color;
	for (int x = draw_rect.left(); x < draw_rect.right(); x++) {
		int real_x = x-contentsRect().x();
		const uint8_t h = screen_x_to_h(real_x);
		PVCore::PVHSVColor(h).toQColor(color);
		painter.fillRect(QRect(x, 0, 1, height), color);
	}

	// Draw the cross
	painter.fillRect(QRect(_cross.x() - CROSS_WIDTH/2, _cross.y() - CROSS_THICK/2, CROSS_WIDTH, CROSS_THICK), Qt::black);
	painter.fillRect(QRect(_cross.x() - CROSS_THICK/2, _cross.y() - CROSS_WIDTH/2, CROSS_THICK, CROSS_WIDTH), Qt::black);

	QFrame::paintEvent(event);
}

void PVWidgets::PVColorPicker::update_h(uint8_t h)
{
	_c = h;
	emit color_changed(_c.h());
}

void PVWidgets::PVColorPicker::set_cross_pos(QPoint const& p)
{
	_cross = p;
	_cross_y_rel = (double)p.y()/(double)size().height();
	update();
}
