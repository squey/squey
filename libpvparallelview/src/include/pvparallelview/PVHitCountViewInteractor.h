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

#ifndef PVPARALLELVIEW_PVHITCOUNTVIEWINTERACTOR_H
#define PVPARALLELVIEW_PVHITCOUNTVIEWINTERACTOR_H

#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>

namespace PVParallelView
{

class PVHitCountView;

class PVHitCountViewInteractor : public PVZoomableDrawingAreaInteractor
{
  public:
	explicit PVHitCountViewInteractor(PVWidgets::PVGraphicsView* parent = nullptr);

	bool resizeEvent(PVZoomableDrawingArea* zda, QResizeEvent*) override;

	bool keyPressEvent(PVZoomableDrawingArea* zda, QKeyEvent* event) override;

	bool wheelEvent(PVZoomableDrawingArea* zda, QWheelEvent* event) override;

  protected:
	static PVHitCountView* get_hit_count_view(PVZoomableDrawingArea* zda);
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVHITCOUNTVIEWINTERACTOR_H
