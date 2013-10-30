/**
 * \file PVInputTypeMenuEntries.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvguiqt/PVInputTypeMenuEntries.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/core/PVClassLibrary.h>

#include <QPushButton>

/**
 * a map function to iterate over input plugins according to their sorted ::internal_name().
 */
template<typename F>
static void map_input_by_sorted_internal_name(const F &f)
{
	LIB_CLASS(PVRush::PVInputType) &input_types = LIB_CLASS(PVRush::PVInputType)::get();
	LIB_CLASS(PVRush::PVInputType)::list_classes const& lf = input_types.get_list();

	typedef std::pair<QString, QString> ele_t;
	std::list<ele_t> pairs;

	for(const auto &it: lf) {
		pairs.push_back(std::make_pair(it->internal_name(),it->registered_name()));
	}

	pairs.sort([](const ele_t& a, const ele_t& b) { return a.first.compare(b.first) < 0; });

	for(const auto &it: pairs) {
		f(it.second, lf[it.second]);
	}
}


void PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_menu(QMenu* menu, QObject* parent, const char* slot)
{
	map_input_by_sorted_internal_name([&](const QString &key, const PVRush::PVInputType_p& in) {
			QAction* action = new QAction(in->menu_input_name(), parent);
			action->setData(QVariant(key));
			action->setShortcut(in->menu_shortcut());
			QObject::connect(action, SIGNAL(triggered()), parent, slot);
			menu->addAction(action);
		});
}

void PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_layout(QBoxLayout* layout, QObject* parent, const char* slot)
{
	map_input_by_sorted_internal_name([&](const QString &key, const PVRush::PVInputType_p& in) {
			QAction* action = new QAction(in->menu_input_name(), parent);
			action->setData(QVariant(key));
			action->setShortcut(in->menu_shortcut());
			QObject::connect(action, SIGNAL(triggered()), parent, slot);

			QPushButton* button = new QPushButton(in->menu_input_name());
			button->setIcon(in->icon());
			button->setCursor(in->cursor());

			QObject::connect(button, SIGNAL(clicked()), action, SLOT(trigger()));

			layout->addWidget(button);
		});
}

PVRush::PVInputType_p PVGuiQt::PVInputTypeMenuEntries::input_type_from_action(QAction* action)
{
	QString const& itype = action->data().toString();
	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(itype);
	return in_t;
}
