#ifndef PVWIDGETS_PVCOLORPICKER_H
#define PVWIDGETS_PVCOLORPICKER_H

#include <pvkernel/core/PVHSVColor.h>

#include <QPoint>
#include <QFrame>

namespace PVWidgets {

class PVColorPicker: public QFrame
{
	Q_OBJECT

public:
	PVColorPicker(QWidget* parent = NULL);
	PVColorPicker(PVCore::PVHSVColor const& c, QWidget* parent = NULL);

public:
	uint8_t h_offset() const { return _offset; }
	void set_h_offset(uint8_t const offset) { _offset = offset; }

	PVCore::PVHSVColor color() const { return _c; }
	void set_color(PVCore::PVHSVColor const& c);

public:
	QSize sizeHint() const override;

signals:
	void color_changed(int h);

protected:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void paintEvent(QPaintEvent* event) override;
	void resizeEvent(QResizeEvent* event) override;

private:
	void init();
	uint8_t screen_x_to_h(const int x) const;
	int h_to_x_screen(const uint8_t h) const;
	void process_mouse_event(QMouseEvent* event);
	void set_cross_pos(QPoint const& p);
	void update_h(uint8_t h);

private:
	PVCore::PVHSVColor _c;
	uint8_t _offset;

	QPoint _cross;
	double _cross_y_rel;
};

}

#endif
