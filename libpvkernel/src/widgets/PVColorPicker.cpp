#include <pvkernel/widgets/PVColorPicker.h>
#include <QPainter>
#include <QPaintEvent>

PVWidgets::PVColorPicker::PVColorPicker(PVCore::PVHSVColor const& c, QWidget* parent):
	QWidget(parent),
	_c(c),
	_offset(0)
{
	setFocusPolicy(Qt::StrongFocus);
}


uint8_t PVWidgets::PVColorPicker::screen_x_to_h(const int x) const
{
	uint8_t h = (uint8_t) ((x*color_max)/width);
	h = (h+h_offset())%color_max;
	return h;
}

int PVWidgets::PVColorPicker::h_to_x_screen(const uint8_t h) const
{
}

QSize PVWidgets::PVColorPicker::sizeHint() const
{
	constexpr size_t color_max = ((1<<HSV_COLOR_NBITS_ZONE)*6);
	return QSize(color_max, 10);
}

void PVWidgets::PVColorPicker::paintEvent(QPaintEvent* event)
{
	constexpr int color_max = ((1<<HSV_COLOR_NBITS_ZONE)*6);

	QPainter painter(this);
	QRect const& rect = event->rect();
	const int width  = size().width();
	const int height = size().height();

	QColor color;
	for (int x = rect.left(); x < rect.right(); x++) {
		const uint8_t h = screen_x_to_h(x);
		PVCore::PVHSVColor(h).toQColor(color);
		painter.fillRect(QRect(x, 0, 1, height), color);
	}
}
