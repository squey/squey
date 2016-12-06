/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVSCATTERVIEWPARAMSWIDGET_H
#define PVPARALLELVIEW_PVSCATTERVIEWPARAMSWIDGET_H

#include <QToolBar>

class QSignalMapper;
class QToolButton;

namespace PVParallelView
{

class PVScatterView;

class PVScatterViewParamsWidget : public QToolBar
{
	Q_OBJECT

  public:
	explicit PVScatterViewParamsWidget(PVScatterView* parent);

  public:
	void update_widgets();

  private Q_SLOTS:
	void set_selection_mode(int mode);

  private:
	PVScatterView* parent_sv();

  private:
	QAction* _show_labels;
	QSignalMapper* _sel_mode_signal_mapper;
	QToolButton* _sel_mode_button;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVSCATTERVIEWPARAMSWIDGET_H
