#ifndef PVWIDGETS_PVCOLORPICKER_H
#define PVWIDGETS_PVCOLORPICKER_H

#include <pvkernel/core/PVHSVColor.h>

#include <QPoint>
#include <QWidget>

#include <cassert>

namespace PVWidgets {

class PVColorPicker: public QWidget
{
	Q_OBJECT

public:
	typedef enum {
		SelectionSingle,
		SelectionInterval
	} SelectionMode;

public:
	PVColorPicker(QWidget* parent = NULL);
	PVColorPicker(PVCore::PVHSVColor const& c, QWidget* parent = NULL);

public:
	inline uint8_t x0() const { return _x0; }
	inline void set_x0(uint8_t const x) { assert(x <= PVCore::PVHSVColor::color_max); _x0 = x; }
	inline uint8_t x1() const { return _x1; }
	inline void set_x1(uint8_t const x) { assert(x <= PVCore::PVHSVColor::color_max); _x1 = x; }

	inline SelectionMode selection_mode() const { return _mode; }
	inline void set_selection_mode(SelectionMode const mode) { _mode = mode; }

	inline PVCore::PVHSVColor color() const { return _c; }

	inline PVCore::PVHSVColor interval_left() const { return _c; }
	inline PVCore::PVHSVColor interval_right() const { return _c1; }

	void set_color(PVCore::PVHSVColor const& c);
	void set_interval(PVCore::PVHSVColor const& c0, PVCore::PVHSVColor const& c1);

	bool allow_empty_interval() const { return _allow_empty_interval; }
	void set_allow_empty_interval(bool b) { _allow_empty_interval = b; }

public:
	QSize sizeHint() const override;

signals:
	void color_changed_left(int h);
	void color_changed_right(int h);

protected:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void paintEvent(QPaintEvent* event) override;
	void resizeEvent(QResizeEvent* event) override;

protected:
	inline int x_interval() const { return x1()-x0(); }

private:
	void init();
	uint8_t screen_x_to_h(int x) const;
	int h_to_screen_x(uint8_t h) const;
	void process_mouse_event(QMouseEvent* event);
	void update_h_left(uint8_t h);
	void update_h_right(uint8_t h);

	void draw_up_triangle(int x, QPainter& painter);
	void draw_down_triangle(int x, QPainter& painter);

	bool is_interval_mode() const { return _mode == SelectionInterval; }

private:
	PVCore::PVHSVColor _c;
	PVCore::PVHSVColor _c1;
	uint8_t _x0;
	uint8_t _x1;

	SelectionMode _mode;
	bool _allow_empty_interval;
};

}

#endif
