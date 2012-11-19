/**
 * \file PVCorrelationMenu.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <assert.h>

#include <QAction>
#include <QMessageBox>
#include <QInputDialog>
#include <QVBoxLayout>

#include <pvguiqt/PVCorrelationMenu.h>
#include <pvguiqt/PVAD2GWidget.h>

#include <pvhive/PVCallHelper.h>

#include <picviz/PVRoot.h>
#include <picviz/PVAD2GView.h>


PVGuiQt::PVCorrelationMenu::PVCorrelationMenu(Picviz::PVRoot* root, QWidget* parent  /* = 0 */) :
	QMenu(parent),
	_root(root)
{
	setTitle("&Correlations");
	QAction* enable_correlation_action = addAction("&Enable correlations");
	connect(enable_correlation_action, SIGNAL(toggled(bool)), this, SLOT(enable_correlations(bool)));
	enable_correlation_action->setCheckable(true);
	enable_correlation_action->setChecked(true);

	_action_create_correlation = addAction("&Create new correlation...");
	connect(_action_create_correlation, SIGNAL(triggered(bool)), this, SLOT(create_new_correlation()));

	addSeparator();
	_separator_create_correlation = addSeparator();
}

void PVGuiQt::PVCorrelationMenu::load_correlations()
{
	for (auto correlation : _root->get_correlations()) {
		add_correlation(correlation.get());
	}
}

void PVGuiQt::PVCorrelationMenu::create_new_correlation()
{
	QString name = QInputDialog::getText(this, "New correlation", "Correlation name:", QLineEdit::Normal);
	if (!name.isEmpty()) {
		add_correlation(name);
	}
}

void PVGuiQt::PVCorrelationMenu::add_correlation(const QString & name)
{
	Picviz::PVRoot_sp root_sp = _root->shared_from_this();
	Picviz::PVAD2GView* correlation = PVHive::call<FUNC(Picviz::PVRoot::add_correlation)>(root_sp, name);
	add_correlation(correlation);
	show_correlation(correlation);
}

void PVGuiQt::PVCorrelationMenu::add_correlation(Picviz::PVAD2GView* correlation)
{
	QMenu* correlation_sub_menu = new QMenu(correlation->get_name());
	insertMenu(_separator_create_correlation, correlation_sub_menu);

	QAction* show_action = correlation_sub_menu->addAction(tr("Show"));
	connect(show_action, SIGNAL(triggered(bool)), this, SLOT(show_correlation()));

	QAction* delete_action = correlation_sub_menu->addAction(tr("Delete"));
	connect(delete_action, SIGNAL(triggered(bool)), this, SLOT(delete_correlation()));

	correlation_sub_menu->menuAction()->setData(qVariantFromValue((void*) correlation));
}

void PVGuiQt::PVCorrelationMenu::show_correlation()
{
	QAction* correlation_show_action = (QAction*) sender();
	assert(correlation_show_action);
	QAction* correlation_action = ((QMenu* )correlation_show_action->parentWidget())->menuAction();
	assert(correlation_action);

	Picviz::PVAD2GView* correlation = (Picviz::PVAD2GView*) correlation_action->data().value<void*>();

	show_correlation(correlation);
}

void PVGuiQt::PVCorrelationMenu::show_correlation(Picviz::PVAD2GView* correlation)
{
	QDialog* ad2g_dialog = new QDialog();
	ad2g_dialog->setWindowTitle(tr("Correlations"));
	PVGuiQt::PVAD2GWidget* ad2g_w = new PVGuiQt::PVAD2GWidget(correlation->shared_from_this(), *_root);
	QVBoxLayout* l = new QVBoxLayout();
	l->addWidget(ad2g_w);
	ad2g_dialog->setLayout(l);
	ad2g_dialog->exec();
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
		Picviz::PVAD2GView* correlation = (Picviz::PVAD2GView*) correlation_action->data().value<void*>();
		Picviz::PVRoot_sp root_sp = _root->shared_from_this();
		PVHive::call<FUNC(Picviz::PVRoot::delete_correlation)>(root_sp, correlation->shared_from_this());
		removeAction(correlation_action);
	}
}

void PVGuiQt::PVCorrelationMenu::enable_correlations(bool enabled)
{
	Picviz::PVRoot_sp root_sp = _root->shared_from_this();
	PVHive::call<FUNC(Picviz::PVRoot::enable_correlations)>(root_sp, enabled);
}
