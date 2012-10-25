/**
 * \file PVInputTypeMenuEntries.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvguiqt/PVInputTypeMenuEntries.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/core/PVClassLibrary.h>

#include <QPushButton>

void PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_menu(QMenu* menu, QObject* parent, const char* slot)
{
	LIB_CLASS(PVRush::PVInputType) &input_types = LIB_CLASS(PVRush::PVInputType)::get();
	LIB_CLASS(PVRush::PVInputType)::list_classes const& lf = input_types.get_list();
	
	LIB_CLASS(PVRush::PVInputType)::list_classes::const_iterator it;

	for (it = lf.begin(); it != lf.end(); it++) {
		PVRush::PVInputType_p in = it.value();
		QAction* action = new QAction(in->menu_input_name(), parent);
		action->setData(QVariant(it.key()));
		action->setShortcut(in->menu_shortcut());
		QObject::connect(action, SIGNAL(triggered()), parent, slot);
		menu->addAction(action);
	}
}

void PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_layout(QBoxLayout* layout, QObject* parent, const char* slot)
{
	LIB_CLASS(PVRush::PVInputType) &input_types = LIB_CLASS(PVRush::PVInputType)::get();
	LIB_CLASS(PVRush::PVInputType)::list_classes const& lf = input_types.get_list();

	LIB_CLASS(PVRush::PVInputType)::list_classes::const_iterator it;

	for (it = lf.begin(); it != lf.end(); it++) {
		PVRush::PVInputType_p in = it.value();
		QAction* action = new QAction(in->menu_input_name(), parent);
		action->setData(QVariant(it.key()));
		action->setShortcut(in->menu_shortcut());
		QObject::connect(action, SIGNAL(triggered()), parent, slot);

		QPushButton* button = new QPushButton(in->menu_input_name());
		button->setIcon(in->icon());
		button->setCursor(in->cursor());

		QObject::connect(button, SIGNAL(clicked()), action, SLOT(trigger()));

		layout->addWidget(button);
	}
}

PVRush::PVInputType_p PVGuiQt::PVInputTypeMenuEntries::input_type_from_action(QAction* action)
{
	QString const& itype = action->data().toString();
	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(itype);
	return in_t;
}
