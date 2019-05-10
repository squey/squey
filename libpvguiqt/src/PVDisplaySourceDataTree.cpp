/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVSource.h>

#include <pvguiqt/PVRootTreeModel.h>
#include <pvguiqt/PVRootTreeView.h>

#include <pvguiqt/PVDisplaySourceDataTree.h>

PVDisplays::PVDisplaySourceDataTree::PVDisplaySourceDataTree()
    : PVDisplaySourceIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::UniquePerParameters,
                        "Data tree",
                        QIcon(":/view-datatree"))
{
}

QWidget* PVDisplays::PVDisplaySourceDataTree::create_widget(Inendi::PVSource* src,
                                                            QWidget* parent,
                                                            Params const&) const
{
	PVGuiQt::PVRootTreeModel* model = new PVGuiQt::PVRootTreeModel(*src);
	PVGuiQt::PVRootTreeView* widget = new PVGuiQt::PVRootTreeView(model, parent);

	return widget;
}
