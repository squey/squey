/* * MIT License
 *
 * © Squey, 2026
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

#ifndef PVPARALLELVIEW_PVFLOATINGTOOLBAR_H
#define PVPARALLELVIEW_PVFLOATINGTOOLBAR_H

#include <pvparallelview/export.h>

#include <QToolBar>

namespace PVParallelView
{

/**
 * A toolbar floating over a view, where no layout resizes it.
 *
 * It fits its contents again whenever they change size -- a colour scheme
 * applied after it was built, or a menu showing a longer name, is enough --
 * rather than leaving the last of them behind the extension button.
 */
class PVPARALLELVIEW_EXPORT PVFloatingToolBar : public QToolBar
{
	Q_OBJECT

  public:
	explicit PVFloatingToolBar(QWidget* parent);

  protected:
	bool event(QEvent* event) override;
};

} // namespace PVParallelView

#endif
