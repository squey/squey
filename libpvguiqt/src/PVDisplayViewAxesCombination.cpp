/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVDisplayViewAxesCombination.h>

#include <inendi/PVView.h>

PVDisplays::PVDisplayViewAxesCombination::PVDisplayViewAxesCombination()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar, "Axes combination")
{
}

QWidget* PVDisplays::PVDisplayViewAxesCombination::create_widget(Inendi::PVView* view,
                                                                 QWidget* parent) const
{
	PVGuiQt::PVAxesCombinationDialog* dlg = new PVGuiQt::PVAxesCombinationDialog(*view, parent);

	return dlg;
}

QIcon PVDisplays::PVDisplayViewAxesCombination::toolbar_icon() const
{
	return QIcon(":/view-layerstack");
}

QString PVDisplays::PVDisplayViewAxesCombination::widget_title(Inendi::PVView* view) const
{
	return "Axes combination [" + QString::fromStdString(view->get_name()) + "]";
}
