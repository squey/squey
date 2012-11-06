#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVDisplayViewAxesCombination.h>

PVDisplays::PVDisplayViewAxesCombination::PVDisplayViewAxesCombination():
	PVDisplayViewIf()
{
}

QWidget* PVDisplays::PVDisplayViewAxesCombination::create_widget(Picviz::PVView* view, QWidget* parent) const
{
	Picviz::PVView_sp view_sp = view->shared_from_this();
	PVGuiQt::PVAxesCombinationDialog* dlg = new PVGuiQt::PVAxesCombinationDialog(view_sp, parent);

	return dlg;
}
