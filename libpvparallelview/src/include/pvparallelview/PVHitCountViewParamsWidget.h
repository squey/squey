/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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
{
	Q_OBJECT

  public:
	explicit PVHitCountViewParamsWidget(PVHitCountView* parent);

  public:
	void update_widgets();

  private Q_SLOTS:
	void set_selection_mode(int mode);

  private:
	PVHitCountView* parent_hcv();

  private:
	QAction* _autofit;
	QAction* _use_log_color;
	QSignalMapper* _signal_mapper;
	QToolButton* _sel_mode_button;
};
}

#endif // PVPARALLELVIEW_PVHITCOUNTVIEWPARAMSWIDGET_H
