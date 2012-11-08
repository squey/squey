/**
 * \file PVCorrelationMenu.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <assert.h>

#include <QMessageBox>
#include <QInputDialog>

#include <pvguiqt/PVCorrelationMenu.h>

PVGuiQt::PVCorrelationMenu::PVCorrelationMenu(QWidget* parent  /* = 0 */) : QMenu(parent)
{
	setTitle("&Correlations");
	QAction* enable_correlation_action = addAction("&Enable correlations");
	connect(enable_correlation_action, SIGNAL(toggled(bool)), this, SIGNAL(correlations_enabled(bool)));
	enable_correlation_action->setCheckable(true);
	enable_correlation_action->setChecked(true);
	_separator_first_correlation = addSeparator();
	_separator_create_correlation = addSeparator();
	_action_create_correlation = addAction("&Create new correlation");
	connect(_action_create_correlation, SIGNAL(triggered(bool)), this, SLOT(create_new_correlation()));
}

void PVGuiQt::PVCorrelationMenu::create_new_correlation()
{
	QString name = QInputDialog::getText(this, "New correlation", "Correlation name:", QLineEdit::Normal);
	if (!name.isEmpty()) {
		add_new_correlation(name);
	}
}

void PVGuiQt::PVCorrelationMenu::add_new_correlation(const QString & name)
{
	QMenu* correlation_sub_menu = new QMenu(name);
	insertMenu(_separator_create_correlation, correlation_sub_menu);

	QAction* show_action = correlation_sub_menu->addAction(tr("Show"));
	connect(show_action, SIGNAL(triggered(bool)), this, SLOT(show_correlation()));

	QAction* delete_action = correlation_sub_menu->addAction(tr("Delete"));
	connect(delete_action, SIGNAL(triggered(bool)), this, SLOT(delete_correlation()));

	emit correlation_added(name);

	show_correlation(get_correlation_index_from_subaction(correlation_sub_menu->menuAction()));
}

void PVGuiQt::PVCorrelationMenu::show_correlation()
{
	QAction* correlation_show_action = (QAction*) sender();
	assert(correlation_show_action);
	QAction* correlation_action = ((QMenu* )correlation_show_action->parentWidget())->menuAction();
	assert(correlation_action);

	show_correlation(get_correlation_index_from_subaction(correlation_action));
}

void PVGuiQt::PVCorrelationMenu::show_correlation(int index)
{
	emit correlation_shown(index);
}

int PVGuiQt::PVCorrelationMenu::get_correlation_index_from_subaction(QAction* action)
{
	assert(action);
	QAction* correlation_action = ((QMenu* )action->parentWidget())->menuAction();
	assert(correlation_action);

	int index = 0;
	for (QAction* a : actions()) {
		if (a == action) {
			break;
		}
		index++;
	}

	int first_action = 0;
	for (QAction* a : actions()) {
		if (a == _separator_first_correlation) {
			break;
		}
		first_action++;
	}

	return index - first_action -1;
}

void PVGuiQt::PVCorrelationMenu::delete_correlation()
{
	QAction* delete_correlation_action = (QAction*) sender();
	assert(delete_correlation_action);
	QAction* correlation_action = ((QMenu* )delete_correlation_action->parentWidget())->menuAction();
	assert(correlation_action);

	QMessageBox::StandardButton ret = QMessageBox::question(
		this,
		tr("Confirm delete"),
		tr("Delete correlation \"%1\"").arg(correlation_action->text()),
		QMessageBox::Yes | QMessageBox::No
	);
	if (ret == QMessageBox::Yes) {
		int index = get_correlation_index_from_subaction(correlation_action);
		removeAction(correlation_action);
		emit correlation_deleted(index);
	}
}
