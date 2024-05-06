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

#ifndef WIDGETS_PVSCALINGMODEWIDGET_H
#define WIDGETS_PVSCALINGMODEWIDGET_H

#include <pvbase/types.h>
#include <pvkernel/widgets/PVComboBox.h>

#include <QPushButton>

namespace Squey
{
class PVScaled;
} // namespace Squey

namespace PVWidgets
{

class PVScalingModeWidget : public QWidget
{
  public:
	explicit PVScalingModeWidget(QWidget* parent = nullptr);
	PVScalingModeWidget(PVCol axis_id, Squey::PVScaled& scaling, QWidget* parent = nullptr);

  public:
	void populate_from_type(QString const& type, QString const& mapped);
	void populate_from_scaling(PVCol axis_id, Squey::PVScaled& scaling);
	inline void select_default() { set_mode("default"); }
	inline void clear() { _combo->clear(); }

	QSize sizeHint() const override;

  public:
	bool set_mode(QString const& mode) { return _combo->select_userdata(mode); }
	inline QString get_mode() const { return _combo->get_sel_userdata().toString(); }

  public:
	PVComboBox* get_combo_box() { return _combo; }

  private:
	PVComboBox* _combo;
};
} // namespace PVWidgets

#endif
