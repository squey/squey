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

#ifndef PVWIDGETS_PVAXISTYPEWIDGET_H
#define PVWIDGETS_PVAXISTYPEWIDGET_H

#include <pvkernel/widgets/PVComboBox.h>

namespace PVWidgets
{

/**
 * This widget is a combo box showing types names.
 */
class PVAxisTypeWidget : public PVComboBox
{
  public:
	/**
	 * Build the combo box with types.
	 */
	explicit PVAxisTypeWidget(QWidget* parent = nullptr);

  public:
	inline QString get_sel_type() const { return currentText(); }
	inline bool sel_type(QString const& type) { return select(type); }
};
} // namespace PVWidgets

#endif
