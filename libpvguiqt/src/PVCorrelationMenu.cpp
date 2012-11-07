/**
 * \file PVCorrelationMenu.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <assert.h>

#include <QMessageBox>

#include <pvguiqt/PVCorrelationMenu.h>

bool PVGuiQt::__impl::CreateNewCorrelationEventFilter::eventFilter(QObject* watched, QEvent* event)
{
	bool rename = false;
	bool close = false;
	if (event->type() == QEvent::Leave) {
		rename = false;
		close = true;
	}
	else if (event->type() == QEvent::KeyPress) {
		QKeyEvent* key_event = (QKeyEvent*) event;
		rename = key_event->key() == Qt::Key_Return;
		close = key_event->key() == Qt::Key_Return || key_event->key() == Qt::Key_Escape;
	}
	if (rename) {
		_menu->add_new_correlation(_line_edit->text());
	}
	if (close) {
		_line_edit->deleteLater();
	}
	return rename;
}

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
}

void PVGuiQt::PVCorrelationMenu::create_new_correlation()
{
	QLineEdit* line_edit = new QLineEdit(this);
	QRect rect = actionGeometry(_action_create_correlation);
	line_edit->move(rect.topLeft());
	line_edit->resize(QSize(rect.width(), rect.height()));
	line_edit->show();
	line_edit->setFocus();
	line_edit->installEventFilter(new __impl::CreateNewCorrelationEventFilter(this, line_edit));
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
}

void PVGuiQt::PVCorrelationMenu::show_correlation()
{
	QAction* correlation_show_action = (QAction*) sender();
	assert(correlation_show_action);
	QAction* correlation_action = ((QMenu* )correlation_show_action->parentWidget())->menuAction();
	assert(correlation_action);

	int index = get_correlation_index_from_subaction(correlation_action);

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
