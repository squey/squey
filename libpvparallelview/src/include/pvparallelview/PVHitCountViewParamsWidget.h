
#ifndef PVPARALLELVIEW_PVHITCOUNTVIEWPARAMSWIDGET_H
#define PVPARALLELVIEW_PVHITCOUNTVIEWPARAMSWIDGET_H

#include <QToolBar>

class QToolBar;
class QCheckBox;
class QSignalMapper;
class QToolButton;

namespace PVParallelView
{

class PVHitCountView;

class PVHitCountViewParamsWidget : public QToolBar
// public PVWidgets::PVConfigPopupWidget
{
Q_OBJECT

public:
	PVHitCountViewParamsWidget(PVHitCountView* parent);

public:
	void update_widgets();

private slots:
	void set_selection_mode(int mode);

private:
	PVHitCountView* parent_hcv();

private:
#if RH_USE_PVConfigPopupWidget
	QCheckBox     *_cb_autofit;
	QCheckBox     *_cb_use_log_color;
#else
	QAction       *_autofit;
	QAction       *_use_log_color;
#endif
	QSignalMapper *_signal_mapper;
	QToolButton   *_sel_mode_button;
};

}

#endif // PVPARALLELVIEW_PVHITCOUNTVIEWPARAMSWIDGET_H
