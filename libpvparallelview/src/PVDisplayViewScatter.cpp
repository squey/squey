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

/*****************************************************************************
 * PVDisplays::PVDisplayViewScatter::PVDisplayViewScatter
 *****************************************************************************/

PVDisplays::PVDisplayViewScatter::PVDisplayViewScatter()
    : PVDisplayViewDataIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCtxtMenu, "Scatter view")
{
}

/*****************************************************************************
 * PVDisplays::PVDisplayViewScatter::create_widget
 *****************************************************************************/

QWidget* PVDisplays::PVDisplayViewScatter::create_widget(Inendi::PVView* view,
                                                         Params const& params,
                                                         QWidget* parent) const
{
	auto axis_x = params.size() > 0 ? params.at(0) : PVCombCol();
	auto axis_y = params.size() > 1 ? params.at(1) : PVCombCol();
	PVParallelView::PVLibView* lib_view = PVParallelView::common::get_lib_view(*view);
	QWidget* widget = lib_view->create_scatter_view(axis_x, axis_y, parent);

	return widget;
}

/*****************************************************************************
 * PVDisplays::PVDisplayViewScatter::toolbar_icon
 *****************************************************************************/

QIcon PVDisplays::PVDisplayViewScatter::toolbar_icon() const
{
	return QIcon(":/view-scatter");
}

/*****************************************************************************
 * PVDisplays::PVDisplayViewScatter::widget_title
 *****************************************************************************/

QString PVDisplays::PVDisplayViewScatter::widget_title(Inendi::PVView* view, Params const&) const
{
	return QString("Scatter view [" + QString::fromStdString(view->get_name()) /* + " on axes '" +
	               view->get_axis_name(axis_comb_x) + "' and '" + view->get_axis_name(axis_comb_y)*/ +
	               "']");
}

/*****************************************************************************
 * PVDisplays::PVDisplayViewScatter::axis_menu_name
 *****************************************************************************/

QString PVDisplays::PVDisplayViewScatter::axis_menu_name(Inendi::PVView* /*view*/,
                                                         Params const&) const
{
	return QString("New scatter view with axis...");
}

void PVDisplays::PVDisplayViewScatter::add_to_axis_menu(QMenu& menu,
                                                        PVCombCol axis_comb,
                                                        Inendi::PVView* view,
                                                        PVDisplays::PVDisplaysContainer* container)
{
	const QStringList& axes = view->get_axes_names_list();
	const QString& view_menu_title = axis_menu_name(
	    view, {axis_comb, view->is_last_axis(axis_comb) ? PVCombCol() : PVCombCol(axis_comb + 1)});
	PVWidgets::PVFilterableMenu* axes_menu =
	    new PVWidgets::PVFilterableMenu(view_menu_title, &menu);
	QList<QAction*> actions;
	QAction* next_axis = nullptr;

	for (PVCombCol i(0); i < view->get_axes_combination().get_axes_count(); i++) {
		if (i != axis_comb) {
			auto create_action = [&]() {
				QAction* act = new QAction();
				act->setText(axes[i]);
				act->connect(act, &QAction::triggered, [container, this, view, axis_comb, i]() {
					container->create_view_axis_widget(*this, view, {axis_comb, PVCombCol(i)});
				});
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
}
