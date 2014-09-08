
#include <pvparallelview/PVZoomedParallelViewParamsWidget.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <QAction>
#include <QMenu>
#include <QToolButton>

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewParamsWidget::PVZoomedParallelViewParamsWidget
 *****************************************************************************/

PVParallelView::PVZoomedParallelViewParamsWidget::PVZoomedParallelViewParamsWidget(PVParallelView::PVZoomedParallelView* parent) :
	QToolBar(parent)
{
	setIconSize(QSize(17, 17));

	_menu_toolbutton = new QToolButton(this);
	addWidget(_menu_toolbutton);
	_menu_toolbutton->setPopupMode(QToolButton::InstantPopup);
	_menu_toolbutton->setIcon(QIcon(":/select-axis"));

	_axes = new QMenu();
	connect(_axes, SIGNAL(triggered(QAction*)),
	        this, SLOT(set_active_axis_action(QAction*)));

	_menu_toolbutton->setMenu(_axes);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewParamsWidget::build_axis_menu
 *****************************************************************************/

void PVParallelView::PVZoomedParallelViewParamsWidget::build_axis_menu(int active_axis,
                                                                       const QStringList &sl)
{
	QAction *current_axis_action;

	_axes->clear();
	int i = 0;
	for (const QString& str : sl) {
		QAction *act = _axes->addAction(str);

		act->setData(i);
		if (i == active_axis) {
			current_axis_action = act;
		}

		++i;
	}

	// resetting active stuff
	_active_axis_action = nullptr;
	_active_axis = -1;

	set_active_axis_action(current_axis_action);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewParamsWidget::set_active_axis_action
 *****************************************************************************/

void PVParallelView::PVZoomedParallelViewParamsWidget::set_active_axis_action(QAction *act)
{
	if (act == _active_axis_action) {
		return;
	}

	if (_active_axis_action) {
		_active_axis_action->setVisible(true);
	}
	act->setVisible(false);
	_active_axis_action = act;

	_menu_toolbutton->setToolTip(QString("Select displayed axis\ncurrent axis is \"") + act->text() + "\"");

	PVCol axis = act->data().toInt();
	if (_active_axis >= 0) {
		/* only if _active_axis has been initialized (after the first
		 * call done by ::build_axis_menu(...))
		 */
		emit change_to_col(axis);
	}
	_active_axis = axis;
}
