
#include <pvparallelview/PVZoomedParallelViewParamsWidget.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <QAction>
#include <QMenu>
#include <QComboBox>

#include <assert.h>

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewParamsWidget::PVZoomedParallelViewParamsWidget
 *****************************************************************************/

PVParallelView::PVZoomedParallelViewParamsWidget::PVZoomedParallelViewParamsWidget(PVParallelView::PVZoomedParallelView* parent) :
	QToolBar(parent)
{
	_combo_box = new QComboBox();
	QAction *action = addWidget(_combo_box);
	action->setVisible(true);
	connect(_combo_box, SIGNAL(activated(int)),
	        this, SLOT(combo_activated(int)));
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewParamsWidget::build_axis_menu
 *****************************************************************************/

void PVParallelView::PVZoomedParallelViewParamsWidget::build_axis_menu(int active_col,
                                                                       const QStringList &sl)
{
	_combo_box->clear();

	int i = 0;
	for (const QString& str : sl) {
		_combo_box->addItem(str);
		++i;
	}
	_combo_box->setCurrentIndex(active_col);
	_active_col = active_col;
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelViewParamsWidget::change_to_axis
 *****************************************************************************/

void PVParallelView::PVZoomedParallelViewParamsWidget::combo_activated(int index)
{
	if (index != _active_col) {
		emit change_to_col(index);
	}
}
