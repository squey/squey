/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/qobject_helpers.h>
#include <pvkernel/core/PVClassLibrary.h>

#include <inendi/PVView.h>

#include <pvdisplays/PVDisplaysImpl.h>
#include <pvdisplays/PVDisplaysContainer.h>

#include <QMenu>
#include <QString>
#include <QAction>

PVDisplays::PVDisplaysImpl* PVDisplays::PVDisplaysImpl::_instance = nullptr;

static const char* plugins_get_displays_dir()
{
	const char* pluginsdir;

	// FIXME : This is dead code
	pluginsdir = getenv("INENDI_DISPLAYS_DIR");

	return pluginsdir;
}

PVDisplays::PVDisplaysImpl& PVDisplays::PVDisplaysImpl::get()
{
	static PVDisplaysImpl instance;
	return instance;
}

void PVDisplays::PVDisplaysImpl::load_plugins()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(
	    QString(plugins_get_displays_dir()), "libdisplay");
	if (ret == 0) {
		PVLOG_WARN("No display plugins have been loaded !\n");
	} else {
		PVLOG_INFO("%d display plugins have been loaded.\n", ret);
	}
}

void PVDisplays::PVDisplayViewDataIf::add_to_axis_menu(QMenu& menu,
                                                       PVCombCol axis_comb,
                                                       Inendi::PVView* view,
                                                       PVDisplays::PVDisplaysContainer* container)
{
	QAction* act = new QAction();
	act->setText(axis_menu_name(view, {axis_comb}));
	act->setIcon(toolbar_icon());
	act->connect(act, &QAction::triggered, [this, view, axis_comb, container]() {
		container->create_view_axis_widget(*this, view, {axis_comb});
	});
	menu.addAction(act);
}

void PVDisplays::PVDisplaysImpl::add_displays_view_axis_menu(QMenu& menu,
                                                             PVDisplaysContainer* container,
                                                             Inendi::PVView* view,
                                                             PVCombCol axis_comb)
{
	visit_displays_by_if<PVDisplayViewDataIf>(
	    [&](PVDisplayViewDataIf& interface) {
		    interface.add_to_axis_menu(menu, axis_comb, view, container);
		},
	    PVDisplayIf::ShowInCtxtMenu);
}
