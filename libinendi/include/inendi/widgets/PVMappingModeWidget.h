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

#ifndef WIDGETS_PVMAPPINGMODEWIDGET_H
#define WIDGETS_PVMAPPINGMODEWIDGET_H

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/widgets/PVComboBox.h>

#include <QWidget>

namespace Inendi
{
class PVMapped;
} // namespace Inendi

namespace PVWidgets
{

class PVMappingModeWidget : public QWidget
{
  public:
	explicit PVMappingModeWidget(QWidget* parent = nullptr);
	PVMappingModeWidget(PVCol axis_id, Inendi::PVMapped& mapping, QWidget* parent = nullptr);

  public:
	void populate_from_type(QString const& type);
	void populate_from_mapping(PVCol axis_id, Inendi::PVMapped& mapping);
	inline void select_default() { set_mode("default"); }
	inline void clear() { _combo->clear(); }

	QSize sizeHint() const override;

  public:
	bool set_mode(QString const& mode);
	inline QString get_mode() const { return _combo->get_sel_userdata().toString(); }

  public:
	PVComboBox* get_combo_box() { return _combo; }

  private:
	PVComboBox* _combo;
};
} // namespace PVWidgets

#endif
