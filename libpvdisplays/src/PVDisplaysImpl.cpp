/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/qobject_helpers.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/widgets/PVFilterableMenu.h>

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
		    const QStringList& axes = view->get_axes_names_list();
		    const QString& view_menu_title = interface.axis_menu_name(
		        view, axis_comb,
		        view->is_last_axis(axis_comb) ? PVCombCol() : PVCombCol(axis_comb + 1));
		    PVWidgets::PVFilterableMenu* axes_menu =
		        new PVWidgets::PVFilterableMenu(view_menu_title, &menu);
		    QList<QAction*> actions;
		    QAction* next_axis = nullptr;

		    for (PVCombCol i(0); i < view->get_axes_combination().get_axes_count(); i++) {
			    if (i != axis_comb) {
				    auto create_action = [&]() {
					    QAction* act =
					        action_bound_to_params(interface, view, axis_comb, PVCombCol(i), false);
					    act->setText(axes[i]);
					    act->setIcon(interface.toolbar_icon());
					    connect(act, SIGNAL(triggered()), receiver, slot);

					    return act;
					};

				    actions << create_action();

				    if (i == (axis_comb + 1)) {
					    next_axis = create_action();
				    }
			    }
		    }

		    axes_menu->addAction(next_axis); // Shortcut for next axis
		    axes_menu->addSeparator();
		    axes_menu->addActions(actions);
		    menu.addMenu(axes_menu);

		},
	    PVDisplayIf::ShowInCtxtMenu);
}

PVDisplays::PVDisplaysContainer*
PVDisplays::PVDisplaysImpl::get_parent_container(QWidget* self) const
{
	return PVCore::get_qobject_parent_of_type<PVDisplaysContainer*>(self);
}
