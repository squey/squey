/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/qobject_helpers.h>
#include <pvkernel/core/PVClassLibrary.h>

#include <pvdisplays/PVDisplaysImpl.h>
#include <pvdisplays/PVDisplaysContainer.h>

#include <QMenu>

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

void PVDisplays::PVDisplaysImpl::add_displays_view_axis_menu(QMenu& menu,
                                                             QObject* receiver,
                                                             const char* slot,
                                                             Inendi::PVView* view,
                                                             PVCombCol axis_comb) const
{
	visit_displays_by_if<PVDisplayViewAxisIf>(
	    [&](PVDisplayViewAxisIf& interface) {
		    QAction* act = action_bound_to_params(interface, view, axis_comb);
		    act->setText(interface.axis_menu_name(view, axis_comb));
		    act->setIcon(interface.toolbar_icon());
		    connect(act, SIGNAL(triggered()), receiver, slot);
		    menu.addAction(act);

		},
	    PVDisplayIf::ShowInCtxtMenu);
}

void PVDisplays::PVDisplaysImpl::add_displays_view_zone_menu(QMenu& menu,
                                                             QObject* receiver,
                                                             const char* slot,
                                                             Inendi::PVView* view,
                                                             PVCombCol axis_comb) const
{
	visit_displays_by_if<PVDisplayViewZoneIf>(
	    [&](PVDisplayViewZoneIf& interface) {
		    QAction* act = action_bound_to_params(interface, view, axis_comb);
		    act->setText(interface.axis_menu_name(view, axis_comb));
		    act->setIcon(interface.toolbar_icon());
		    connect(act, SIGNAL(triggered()), receiver, slot);
		    menu.addAction(act);

		},
	    PVDisplayIf::ShowInCtxtMenu);
}

PVDisplays::PVDisplaysContainer*
PVDisplays::PVDisplaysImpl::get_parent_container(QWidget* self) const
{
	return PVCore::get_qobject_parent_of_type<PVDisplaysContainer*>(self);
}
