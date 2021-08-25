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

#ifndef PVWIDGETS_PVGRAPHICSVIEWINTERACTORSCENE_H
#define PVWIDGETS_PVGRAPHICSVIEWINTERACTORSCENE_H

#include <pvkernel/widgets/PVGraphicsViewInteractor.h>

namespace PVWidgets
{

class PVGraphicsViewInteractorScene : public PVGraphicsViewInteractor<PVGraphicsView>
{
  public:
	explicit PVGraphicsViewInteractorScene(PVGraphicsView* parent);

  protected:
	bool contextMenuEvent(PVGraphicsView* obj, QContextMenuEvent* event) override;

	bool mouseDoubleClickEvent(PVGraphicsView* obj, QMouseEvent* event) override;
	bool mousePressEvent(PVGraphicsView* obj, QMouseEvent* event) override;
	bool mouseReleaseEvent(PVGraphicsView* obj, QMouseEvent* event) override;
	bool mouseMoveEvent(PVGraphicsView* obj, QMouseEvent* event) override;

	bool wheelEvent(PVGraphicsView* obj, QWheelEvent* event) override;

	bool keyPressEvent(PVGraphicsView* obj, QKeyEvent* event) override;
	bool keyReleaseEvent(PVGraphicsView* obj, QKeyEvent* event) override;
};
} // namespace PVWidgets

#endif // PVWIDGETS_PVGRAPHICSVIEWINTERACTORSCENE_H
