#ifndef PVWIDGETS_PVCOLORDIALOG_H
#define PVWIDGETS_PVCOLORDIALOG_H

#include "../../../ui_PVColorDialog.h"

#include <pvkernel/core/PVHSVColor.h>

namespace PVCore {
class PVHSVColor;
}

namespace PVWidgets {

class PVColorPicker;

class PVColorDialog: public QDialog, Ui::PVColorDialog
{
	Q_OBJECT

public:
	PVColorDialog(QWidget* parent = NULL);
	PVColorDialog(PVCore::PVHSVColor const& c, QWidget* parent = NULL);

public:
	void set_color(PVCore::PVHSVColor const c);
	inline PVCore::PVHSVColor color() const { return picker()->color(); };

	inline void set_interval(uint8_t x0, uint8_t x1)
	{
		assert(x1 > x0);
		picker()->set_x0(x0);
		picker()->set_x1(x1);
	}

private:
	void show_color(PVCore::PVHSVColor const c);

private slots:
	void picker_color_changed(int h);

signals:
	void color_changed(int h);

private:
	void init();

	inline PVColorPicker* picker() { return _picker; }
	inline PVColorPicker const* picker() const { return _picker; }
};

}

#endif
