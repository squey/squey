/* * MIT License
 *
 * © Squey, 2024
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <pvguiqt/PVDisplayViewFilters.h>

#include <squey/PVView.h>
#include <squey/PVLayerFilter.h>

#include <pvguiqt/PVLayerFilterProcessWidget.h>

#include <pvkernel/widgets/PVModdedIcon.h>
#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>

#define ARG_NAME_AXIS "axis"
#define ARG_DESC_AXIS "Axis"

PVDisplays::PVDisplayViewFilters::PVDisplayViewFilters()
    : PVDisplayViewIf(ShowInCtxtMenu,
                      "Filters",
                      PVModdedIcon("mapping-scaling"))
{
}

auto PVDisplays::PVDisplayViewFilters::create_widget(Squey::PVView*, QWidget* parent, Params const&) const -> QWidget*
{
	assert("Should never get here");
	return new QWidget(parent);
}

// Check if we have already a menu with this name at this level
static QMenu* create_filters_menu_exists(QHash<QMenu*, int> actions_list, QString name, int level)
{
	QHashIterator<QMenu*, int> iter(actions_list);
	while (iter.hasNext()) {
		iter.next();
		QString menu_title = iter.key()->title();
		int menu_level = iter.value();

		if ((!menu_title.compare(name)) && (menu_level == level)) {
			return iter.key();
		}
	}

	return nullptr;
}

void PVDisplays::PVDisplayViewFilters::add_to_axis_menu(
	QMenu& menu, PVCol axis, PVCombCol,
	Squey::PVView* view, PVDisplaysContainer*)
{
	QMenu* filters_menu = menu.addMenu("Filters");
	filters_menu->setAttribute(Qt::WA_TranslucentBackground);
	filters_menu->setIcon(PVModdedIcon("filters"));
	menu.addSeparator();

	QHash<QMenu*, int> actions_list; // key = action name; value = menu level;
	                                 // Foo/Bar/Camp makes Foo at level 0, Bar at
	                                 // level 1, etc.

	LIB_CLASS(Squey::PVLayerFilter)& filters_layer = LIB_CLASS(Squey::PVLayerFilter)::get();
	LIB_CLASS(Squey::PVLayerFilter)::list_classes const& lf = filters_layer.get_list();
	LIB_CLASS(Squey::PVLayerFilter)::list_classes::const_iterator it;

	auto filter_func = [=](const QString& filter_name) {
		Squey::PVLayerFilter::p_type filter_org =
			LIB_CLASS(Squey::PVLayerFilter)::get().get_class_by_name(filter_name);
		Squey::PVLayerFilter::p_type fclone = filter_org->clone<Squey::PVLayerFilter>();
		PVCore::PVArgumentList& args = view->get_last_args_filter(filter_name);
		args[PVCore::PVArgumentKey(ARG_NAME_AXIS, QObject::tr(ARG_DESC_AXIS))].setValue(PVCore::PVOriginalAxisIndexType(axis));
		auto* filter_widget = new PVGuiQt::PVLayerFilterProcessWidget(view, args, fclone);
		filter_widget->show();
	};

	for (it = lf.begin(); it != lf.end(); it++) {
		//(*it).get_args()["Menu_name"]
		QString filter_name = it->key();
		QString action_name = it->value()->menu_name();
		QString status_tip = it->value()->status_bar_description();

		QStringList actions_name = action_name.split(QString("/"));
		if (actions_name.count() > 1) {
			// // qDebug("actions_name[0]=%s\n", qPrintable(actions_name[0]));
			// // We add the various submenus
			for (int i = 0; i < actions_name.count(); i++) {
				bool is_last = i == actions_name.count() - 1 ? true : false;

				// Step 1: we add the different menus into the hash
				QMenu* menu_exists = create_filters_menu_exists(actions_list, actions_name[i], i);
				if (!menu_exists) {
					auto* filter_element_menu = new QMenu(actions_name[i]);
					actions_list[filter_element_menu] = i;
				}

				// Step 2: we connect the menus with each other and connect the actions
				QMenu* menu_to_add = create_filters_menu_exists(actions_list, actions_name[i], i);
				menu_to_add->setAttribute(Qt::WA_TranslucentBackground);
				if (!menu_to_add) {
					PVLOG_ERROR("The menu named '%s' at position level %d cannot be "
					            "added since it was not append previously!\n",
					            qPrintable(actions_name[i]), i);
				}
				if (i == 0) { // We are at root level
					filters_menu->addMenu(menu_to_add);
				} else {
					if (is_last) {
						QMenu* previous_menu =
						    create_filters_menu_exists(actions_list, actions_name[i - 1], i - 1);

						auto* action = new QAction(actions_name[i] + "...", previous_menu);
						action->setObjectName(filter_name);
						action->setStatusTip(status_tip);
						QObject::connect(action, &QAction::triggered, [=](){ filter_func(filter_name); });
						previous_menu->addAction(action);
					} else {
						// we add a menu to the previous menu
						QMenu* previous_menu =
						    create_filters_menu_exists(actions_list, actions_name[i - 1], i - 1);
						QMenu* current_menu =
						    create_filters_menu_exists(actions_list, actions_name[i], i);
						previous_menu->addMenu(current_menu);
					}
				}
			}
		} else { // Nothing to split, so there is only a direct action
			auto* action = new QAction(action_name + "...", &menu);
			action->setObjectName(filter_name);
			action->setStatusTip(status_tip);
			QObject::connect(action, &QAction::triggered, [=](){ filter_func(filter_name); });

			filters_menu->addAction(action);
		}
	}



}
