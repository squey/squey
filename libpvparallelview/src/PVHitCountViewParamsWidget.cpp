
#include <pvparallelview/PVHitCountViewParamsWidget.h>
#include <pvparallelview/PVHitCountView.h>

#include <QVBoxLayout>
#include <QCheckBox>

/*****************************************************************************
 * PVParallelView::PVHitCountViewParamsWidget::PVHitCountViewParamsWidget
 *****************************************************************************/

PVParallelView::PVHitCountViewParamsWidget::PVHitCountViewParamsWidget(PVHitCountView* parent) :
	PVWidgets::PVConfigPopupWidget(parent)
{
	setWindowTitle(tr("Hit count view - options"));

	_cb_autofit = new QCheckBox(tr("Auto-fit selection on the occurence axis"));
	_cb_use_log_color = new QCheckBox(tr("Use logarithmic colormap"));

	QVBoxLayout* layout = new QVBoxLayout();
	layout->addWidget(_cb_autofit);
	layout->addWidget(_cb_use_log_color);
	setContentLayout(layout);

	connect(_cb_autofit, SIGNAL(toggled(bool)),
	        parent_hcv(), SLOT(toggle_auto_x_zoom_sel()));
	connect(_cb_use_log_color,  SIGNAL(toggled(bool)),
	        parent_hcv(), SLOT(toggle_log_color()));
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewParamsWidget::update_widgets
 *****************************************************************************/

void PVParallelView::PVHitCountViewParamsWidget::update_widgets()
{
	_cb_autofit->blockSignals(true);
	_cb_use_log_color->blockSignals(true);

	_cb_autofit->setChecked(parent_hcv()->auto_x_zoom_sel());
	_cb_use_log_color->setChecked(parent_hcv()->use_log_color());

	_cb_autofit->blockSignals(false);
	_cb_use_log_color->blockSignals(false);
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewParamsWidget::parent_hcv
 *****************************************************************************/

PVParallelView::PVHitCountView* PVParallelView::PVHitCountViewParamsWidget::parent_hcv()
{
	assert(qobject_cast<PVHitCountView*>(parentWidget()));
	return static_cast<PVHitCountView*>(parentWidget());
}

