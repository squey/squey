#ifndef PVWIDGETS_PVCOLORPICKER_H
#define PVWIDGETS_PVCOLORPICKER_H

#include <pvkernel/core/PVHSVColor.h>
#include <QWidget>

namespace PVWidgets {

class PVColorPicker: public QWidget
{
	Q_OBJECT

public:
	PVColorPicker(PVCore::PVHSVColor const& c = PVCore::PVHSVColor(0), QWidget* parent = NULL);

public:
	uint8_t h_offset() const { return _offset; }
	void set_h_offset(uint8_t const offset) { _offset = offset; }

public:
	QSize sizeHint() const override;

protected:
	void paintEvent(QPaintEvent* event) override;

private:
	uint8_t screen_x_to_h(const int x) const;
	int h_to_x_screen(const uint8_t h) const;

private:
	PVCore::PVHSVColor _c;
	uint8_t _offset;
};

}

#endif
