/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVWIDGETS_PVCOLORDIALOG_H
#define PVWIDGETS_PVCOLORDIALOG_H

#include <pvkernel/core/PVHSVColor.h>
#include <pvkernel/ui_PVColorDialog.h>

namespace PVCore
{
class PVHSVColor;
} // namespace PVCore

namespace PVWidgets
{

namespace __impl
{
class PVLabelEventFilter;
} // namespace __impl

class PVColorPicker;

class PVColorDialog : public QDialog, Ui::PVColorDialog
{
	Q_OBJECT

	friend class __impl::PVLabelEventFilter;

  public:
	explicit PVColorDialog(QWidget* parent = nullptr);
	explicit PVColorDialog(PVCore::PVHSVColor const& c, QWidget* parent = nullptr);

  public:
	void set_color(PVCore::PVHSVColor const& c);
	inline PVCore::PVHSVColor color() const { return picker()->color(); };

	inline void set_interval(uint8_t x0, uint8_t x1)
	{
		assert(x1 > x0);
		picker()->set_x0(x0);
		picker()->set_x1(x1);
	}

  Q_SIGNALS:
	void color_changed(int h);

  protected:
	void label_button_pressed(QLabel* label, QMouseEvent* event);
	void label_button_released(QLabel* label, QMouseEvent* event);

  private:
	void show_color(PVCore::PVHSVColor const& c);

	inline PVColorPicker* picker() { return _picker; }
	inline PVColorPicker const* picker() const { return _picker; }

	void set_predefined_color_from_label(QLabel* label);
	void unselect_all_preselected_colors();

  private Q_SLOTS:
	void picker_color_changed(int h);
	void set_predefined_color_from_action();
	void reset_predefined_color_from_action();

  private:
	__impl::PVLabelEventFilter* _label_event_filter;
};
} // namespace PVWidgets

#endif
