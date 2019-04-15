/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H

#include <pvkernel/widgets/PVGraphicsView.h>

class QPainter;
class QStyleOptionGraphicsItem;
class QWidget;

namespace Inendi
{
class PVAxesCombination;
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
	explicit PVZoomedParallelView(Inendi::PVAxesCombination const& axes_comb,
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

	void set_displayed_axis_name(const QString& s) { _display_axis_name = s; }

	/**
	 *
	 */
	void drawForeground(QPainter* painter, const QRectF& rect) override;

  protected:
	PVWidgets::PVHelpWidget* help_widget() { return _help_widget; }
	PVZoomedParallelViewParamsWidget* params_widget() { return _params_widget; }

  private:
	QString _display_axis_name;
	PVWidgets::PVHelpWidget* _help_widget;
	PVZoomedParallelViewParamsWidget* _params_widget;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
