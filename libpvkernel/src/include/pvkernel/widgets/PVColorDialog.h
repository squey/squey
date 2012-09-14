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
