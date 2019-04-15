/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELVIEWPARAMSWIDGET_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELVIEWPARAMSWIDGET_H

#include <QToolBar>

#include <pvbase/types.h>
#include <inendi/widgets/PVAxisComboBox.h>

class QStringList;
class QMenu;
class QToolButton;

namespace PVParallelView
{
class PVZoomedParallelView;

class PVZoomedParallelViewParamsWidget : public QToolBar
{
	Q_OBJECT

  public:
	explicit PVZoomedParallelViewParamsWidget(Inendi::PVAxesCombination const& axes_comb,
	                                          QWidget* parent);

  public:
	void build_axis_menu(PVCombCol active_axis);

  Q_SIGNALS:
	void change_to_col(PVCombCol new_axis);

  private:
	PVWidgets::PVAxisComboBox* _menu;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_ZOOMEDPARALLELVIEWPARAMSWIDGET_H
