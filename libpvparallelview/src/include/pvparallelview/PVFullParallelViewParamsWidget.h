/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef PVPARALLELVIEW_PVFULLPARALLELVIEWPARAMSWIDGET_H
#define PVPARALLELVIEW_PVFULLPARALLELVIEWPARAMSWIDGET_H

#include <QToolBar>

class QToolBar;
class QCheckBox;
class QSignalMapper;
class QToolButton;

namespace PVParallelView
{

class PVFullParallelView;

class PVFullParallelViewParamsWidget : public QToolBar
{
	Q_OBJECT

  public:
	explicit PVFullParallelViewParamsWidget(PVFullParallelView* parent);

  public:
	void update_widgets();

  private Q_SLOTS:
	void set_selection_mode(int mode);

  private:
	PVFullParallelView* parent_hcv();

  private:
	QAction* _autofit;
	QAction* _use_log_color;
	QAction* _show_labels;
	QSignalMapper* _signal_mapper;
	QToolButton* _sel_mode_button;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVFULLPARALLELVIEWPARAMSWIDGET_H
