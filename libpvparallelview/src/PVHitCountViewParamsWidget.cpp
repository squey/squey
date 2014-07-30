

#include <pvparallelview/PVHitCountViewParamsWidget.h>
#include <pvparallelview/PVHitCountViewSelectionRectangle.h>
#include <pvparallelview/PVHitCountView.h>

#include <QVBoxLayout>
#include <QToolBar>
#include <QCheckBox>
#include <QSignalMapper>
#include <QMenu>

/*****************************************************************************
 * PVParallelView::PVHitCountViewParamsWidget::PVHitCountViewParamsWidget
 *****************************************************************************/

PVParallelView::PVHitCountViewParamsWidget::PVHitCountViewParamsWidget(PVHitCountView* parent) :
	QToolBar(parent)
	//	PVWidgets::PVConfigPopupWidget(parent)
{
#if RH_USE_PVConfigPopupWidget
	setWindowTitle(tr("Hit count view - options"));

	QVBoxLayout* layout = new QVBoxLayout();
	setContentLayout(layout);

	_signal_mapper = new QSignalMapper(this);
	QObject::connect(_signal_mapper, SIGNAL(mapped(int)),
	                 this, SLOT(set_selection_mode(int)));

	_toolbar = new QToolBar("Selection mode");
	_toolbar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	_toolbar->setOrientation(Qt::Vertical);
	_toolbar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
	_toolbar->setFloatable(false);
	_toolbar->setMovable(false);

	_toolbar->setContentsMargins(0, 0, 0, 0);
	_toolbar->layout()->setAlignment(Qt::AlignLeft);
	layout->addWidget(_toolbar);

	PVSelectionRectangle::add_selection_mode_selector(parent,
	                                                  _toolbar,
	                                                  _signal_mapper);

	QLayout *l = _toolbar->layout();
	for (int i = 0; i < l->count(); ++i) {
		QLayoutItem* item = l->itemAt(i);
		item->setAlignment(Qt::AlignLeft);
		QWidget *w = item->widget();
		if (w != nullptr) {
			w->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
		}
	}

	_toolbar->addSeparator();

	_cb_autofit = new QCheckBox(tr("Auto-fit selection on the occurence axis"));
	_cb_use_log_color = new QCheckBox(tr("Use logarithmic colormap"));

	_toolbar->addWidget(_cb_autofit);
	_toolbar->addWidget(_cb_use_log_color);

	connect(_cb_autofit, SIGNAL(toggled(bool)),
	        parent_hcv(), SLOT(toggle_auto_x_zoom_sel()));
	connect(_cb_use_log_color,  SIGNAL(toggled(bool)),
	        parent_hcv(), SLOT(toggle_log_color()));
#else // is a QToolBar
	_signal_mapper = new QSignalMapper(this);
	QObject::connect(_signal_mapper, SIGNAL(mapped(int)),
	                 this, SLOT(set_selection_mode(int)));

	_sel_mode_button = PVSelectionRectangle::add_selection_mode_selector(parent,
	                                                                     this,
	                                                                     _signal_mapper);

	addSeparator();

	_autofit = new QAction(this);
	_autofit->setIcon(QIcon(":/zoom-autofit-horizontal"));
	_autofit->setCheckable(true);
	_autofit->setChecked(false);
	_autofit->setShortcut(Qt::Key_F);
	_autofit->setText("View auto-fit on selected events");
	_autofit->setToolTip("Activate/deactivate horizontal auto-fit on selected events ("
	                    + _autofit->shortcut().toString() + ")");
	addAction(_autofit);
	parent->addAction(_autofit);
	connect(_autofit, SIGNAL(toggled(bool)),
	        parent_hcv(), SLOT(toggle_auto_x_zoom_sel()));

	_use_log_color = new QAction("Logarithmic colormap", this);
	_use_log_color->setIcon(QIcon(":/colormap-log"));
	_use_log_color->setCheckable(true);
	_use_log_color->setChecked(false);
	_use_log_color->setShortcut(Qt::Key_S);
	_use_log_color->setText("Logarithmic colormap");
	_use_log_color->setToolTip("Activate/deactivate use of a logarithmic colormap for visible events ("
	                           + _use_log_color->shortcut().toString() + ")");
	addAction(_use_log_color);
	parent->addAction(_use_log_color);
	connect(_use_log_color, SIGNAL(toggled(bool)),
	        parent_hcv(), SLOT(toggle_log_color()));
#endif
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewParamsWidget::update_widgets
 *****************************************************************************/

void PVParallelView::PVHitCountViewParamsWidget::update_widgets()
{
#if RH_USE_PVConfigPopupWidget
	_cb_autofit->blockSignals(true);
	_cb_use_log_color->blockSignals(true);

	_cb_autofit->setChecked(parent_hcv()->auto_x_zoom_sel());
	_cb_use_log_color->setChecked(parent_hcv()->use_log_color());

	_cb_autofit->blockSignals(false);
	_cb_use_log_color->blockSignals(false);
#else
	_autofit->blockSignals(true);
	_use_log_color->blockSignals(true);

	_autofit->setChecked(parent_hcv()->auto_x_zoom_sel());
	_use_log_color->setChecked(parent_hcv()->use_log_color());

	_autofit->blockSignals(false);
	_use_log_color->blockSignals(false);
#endif
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewParamsWidget::set_selection_mode
 *****************************************************************************/

void PVParallelView::PVHitCountViewParamsWidget::set_selection_mode(int mode)
{
	PVHitCountView *hcv = parent_hcv();
	hcv->get_selection_rect()->set_selection_mode(mode);
	hcv->fake_mouse_move();
	hcv->get_viewport()->update();

	PVSelectionRectangle::update_selection_mode_selector(_sel_mode_button,
	                                                     mode);
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewParamsWidget::parent_hcv
 *****************************************************************************/

PVParallelView::PVHitCountView* PVParallelView::PVHitCountViewParamsWidget::parent_hcv()
{
	assert(qobject_cast<PVHitCountView*>(parentWidget()));
	return static_cast<PVHitCountView*>(parentWidget());
}

