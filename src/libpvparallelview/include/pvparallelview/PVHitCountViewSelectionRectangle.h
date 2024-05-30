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

#ifndef PVPARALLELVIEW_PVHITCOUNTVIEWSELECTIONRECTANGLE_H
#define PVPARALLELVIEW_PVHITCOUNTVIEWSELECTIONRECTANGLE_H

#include <pvparallelview/PVSelectionRectangle.h>

namespace Squey
{

class PVView;
} // namespace Squey

namespace PVParallelView
{

class PVHitCountView;

/**
 * @class PVHitCountViewSelectionRectangle
 *
 * a selection rectangle usable with a hit-count view.
 */
class PVHitCountViewSelectionRectangle : public PVParallelView::PVSelectionRectangle
{
  public:
	/**
	 * create a selection rectangle for hit-count view
	 *
	 * @param hcv the "parent" hit-count view
	 */
	explicit PVHitCountViewSelectionRectangle(PVHitCountView* hcv);

  protected:
	/**
	 * selection commit for hit-count view
	 *
	 * @param use_selection_modifiers
	 */
	void commit(bool use_selection_modifiers) override;

  private:
	PVHitCountView* _hcv;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVHITCOUNTVIEWSELECTIONRECTANGLE_H
