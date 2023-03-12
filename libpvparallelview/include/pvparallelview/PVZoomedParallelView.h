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

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H

#include <pvkernel/widgets/PVGraphicsView.h>
#include <pvbase/types.h>

class QPainter;
class QStyleOptionGraphicsItem;
class QWidget;

namespace Squey
{
class PVAxesCombination;
class PVView;
}

namespace PVWidgets
{

class PVHelpWidget;
} // namespace PVWidgets

namespace PVParallelView
{

// forward declaration
class PVZoomedParallelScene;
class PVZoomedParallelViewParamsWidget;

/**
 * @class PVZoomedParallelView
 *
 * A derived class of QGraphicsView to use when displaying a zoom view of parallel coordinates
 * representation of events.
 */
class PVZoomedParallelView : public PVWidgets::PVGraphicsView
{
	friend class PVZoomedParallelScene;

  public:
	/**
	 * Constructor
	 *
	 * @param parent parent widget
	 */
	explicit PVZoomedParallelView(Squey::PVAxesCombination const& axes_comb,
	                              QWidget* parent = nullptr);

	/**
	 * Destructor
	 */
	~PVZoomedParallelView() override;

	/**
	 * Overload method when a resize event occurs.
	 *
	 * @param event then resize event
	 */
	void resizeEvent(QResizeEvent* event) override;

	void update_window_title(Squey::PVView& view, PVCombCol combcol);

  protected:
	PVWidgets::PVHelpWidget* help_widget() { return _help_widget; }
	PVZoomedParallelViewParamsWidget* params_widget() { return _params_widget; }

  private:
	PVWidgets::PVHelpWidget* _help_widget;
	PVZoomedParallelViewParamsWidget* _params_widget;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
