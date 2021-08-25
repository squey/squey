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

#ifndef PVWIDGETS_PVHELPWIDGET_H
#define PVWIDGETS_PVHELPWIDGET_H

#include <pvkernel/widgets/PVTextPopupWidget.h>

namespace PVWidgets
{

class PVHelpWidget : public PVTextPopupWidget
{
  public:
	explicit PVHelpWidget(QWidget* parent);

	/**
	 * test if key is one of the close keys
	 *
	 * @param key the key to test
	 *
	 * @return true if key is one of the help key; false otherwise
	 *
	 * @see PVWidgets::PVHelpWidget*
	 */
	bool is_close_key(int key) override;

  public:
	/**
	 * test if key is one of the help keys
	 *
	 * @param key the key to test
	 *
	 * @return true if key is one of the help key; false otherwise
	 */
	static bool is_help_key(int key);
};
} // namespace PVWidgets

#endif // PVWIDGETS_PVHELPWIDGET_H
