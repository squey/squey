/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvparallelview/PVDisplayViewScatter.h>

#include <pvdisplays/PVDisplaysContainer.h>
#include <pvkernel/widgets/PVFilterableMenu.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVScatterView.h>

PVDisplays::PVDisplayViewScatter::PVDisplayViewScatter()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCtxtMenu, "Scatter view")
{
}

QWidget* PVDisplays::PVDisplayViewScatter::create_widget(Inendi::PVView* view,
                                                         QWidget* parent,
                                                         Params const& params) const
{
	PVParallelView::PVLibView* lib_view = PVParallelView::common::get_lib_view(*view);
	QWidget* widget = lib_view->create_scatter_view(col_param(view, params, 0),
	                                                col_param(view, params, 1), parent);

	return widget;
}

QIcon PVDisplays::PVDisplayViewScatter::toolbar_icon() const
{
	return QIcon(":/view-scatter");
}

QString PVDisplays::PVDisplayViewScatter::widget_title(Inendi::PVView* view) const
{
	return QString("Scatter view [" + QString::fromStdString(view->get_name()) /* + " on axes '" +
	               view->get_axis_name(axis_comb_x) + "' and '" + view->get_axis_name(axis_comb_y)*/ +
	               "']");
}

QString PVDisplays::PVDisplayViewScatter::axis_menu_name(Inendi::PVView*) const
{
	return QString("New scatter view with axis...");
}

void PVDisplays::PVDisplayViewScatter::add_to_axis_menu(QMenu& menu,
                                                        PVCol axis,
                                                        PVCombCol axis_comb,
                                                        Inendi::PVView* view,
                                                        PVDisplays::PVDisplaysContainer* container)
{
	const QStringList& axes_names = view->get_axes_names_list();
	const QString& view_menu_title = axis_menu_name(view);

	if (axis_comb == PVCombCol()) {
		auto act = new QAction(toolbar_icon(), axis_menu_name(view));
		act->connect(act, &QAction::triggered, [container, this, view, axis]() {
			container->create_view_widget(*this, view, {axis, PVCol()});
		});
		menu.addAction(act);
		return;
	}

	PVWidgets::PVFilterableMenu* axes_menu =
	    new PVWidgets::PVFilterableMenu(view_menu_title, &menu);
	QList<QAction*> actions;
	QAction* next_axis = nullptr;

	for (PVCombCol i(0); i < view->get_axes_combination().get_axes_count(); i++) {
		if (i != axis_comb) {
			auto create_action = [&]() {
				QAction* act = new QAction(axes_names[i]);
				act->connect(act, &QAction::triggered, [container, this, view, axis_comb, i]() {
					container->create_view_widget(*this, view, {axis_comb, PVCombCol(i)});
				});
				return act;
			};

			actions << create_action();

			if (i == (axis_comb + 1)) {
				next_axis = create_action();
			}
		}
	}

	axes_menu->setIcon(toolbar_icon());
	axes_menu->addAction(next_axis); // Shortcut for next axis
	axes_menu->addSeparator();
	axes_menu->addActions(actions);
	menu.addMenu(axes_menu);
}
