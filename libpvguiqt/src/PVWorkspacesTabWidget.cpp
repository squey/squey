/**
 * \file PVWorkspacesTabWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <picviz/PVSource.h>

#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVWorkspace.h>

#include <iostream>
#include <QApplication>
#include <QEvent>
#include <QTabBar>
#include <QMouseEvent>
#include <QPushButton>
#include <QDateTime>
#include <QPropertyAnimation>

#define AUTOMATIC_TAB_SWITCH_TIMER_MSEC 500

PVGuiQt::PVWorkspacesTabWidget::PVWorkspacesTabWidget(QWidget* parent) :
	QTabWidget(parent),
	_automatic_tab_switch_timer(this)

{
	setObjectName("PVWorkspacesTabWidget");

	// To get notified of mouse events we must enable mouse tracking on *both* QTabWidget and its underlying QTabBar
	setMouseTracking(true);
	tabBar()->setMouseTracking(true);

	// Automatic tab switching handling  for drag&drop
	_automatic_tab_switch_timer.setSingleShot(true);
	connect(&_automatic_tab_switch_timer, SIGNAL(timeout()), this, SLOT(switch_tab()));

	setTabsClosable(true);
	connect(tabBar(), SIGNAL(tabCloseRequested(int)), this, SLOT(tabCloseRequested_Slot(int)));

	QWidget* new_tab = new QWidget();
	QTabWidget::addTab(new_tab, QIcon(":/more.png"), "");
	setTabToolTip(0, tr("New open workspace"));
	QPushButton* hidden_close_button = new QPushButton();
	hidden_close_button->resize(QSize(0, 0));
	tabBar()->setTabButton(0, QTabBar::RightSide, hidden_close_button);
	connect(this, SIGNAL(currentChanged(int)), this, SLOT(tab_changed(int)));
}

int PVGuiQt::PVWorkspacesTabWidget::count() const
{
	return QTabWidget::count() -1; // Substract new workspace special tab from count
}

void PVGuiQt::PVWorkspacesTabWidget::tab_changed(int index)
{
	if (index == count()) {
		addTab(new PVWorkspace(this), "Open workspace");
		setCurrentIndex(count()-1);
	}
}

void PVGuiQt::PVWorkspacesTabWidget::mouseMoveEvent(QMouseEvent* event)
{
	if (tabBar()->tabAt(event->pos()) == count()) {
		setCursor(Qt::PointingHandCursor);
	}
	else {
		setCursor(Qt::ArrowCursor);
	}
}

int PVGuiQt::PVWorkspacesTabWidget::addTab(QWidget* page, const QString & label)
{
	int index = insertTab(count(), page, label);

	QPropertyAnimation *animation = new QPropertyAnimation(this, "tab_size");
	animation->setDuration(200);
	animation->setStartValue(0);
	animation->setEndValue(100);
	animation->start();

	return index;
}

/*void PVGuiQt::PVWorkspacesTabWidget::removeTab(int index)
{
	QPropertyAnimation *animation = new QPropertyAnimation(this, "tab_size");
	animation->setDuration(200);
	animation->setStartValue(100);
	animation->setEndValue(0);
	animation->start();

	QTabWidget::removeTab(index);
}*/

void PVGuiQt::PVWorkspacesTabWidget::set_tab_size(int tab_size_percent)
{

	QString str = QString("QTabBar::tab:selected { width: %1%;}").arg(tab_size_percent);
	if (tab_size_percent % 100 == 0) {
		tabBar()->setStyleSheet("");
	}
	else {
		tabBar()->setStyleSheet(str);
	}
}

void PVGuiQt::PVWorkspacesTabWidget::tabInserted(int index)
{
	connect(widget(index), SIGNAL(try_automatic_tab_switch()), this, SLOT(start_checking_for_automatic_tab_switch()));
	QTabWidget::tabInserted(index);
}

void PVGuiQt::PVWorkspacesTabWidget::start_checking_for_automatic_tab_switch()
{
	QPoint mouse_pos = tabBar()->mapFromGlobal(QCursor::pos());
	_tab_index = tabBar()->tabAt(mouse_pos);

	if (_tab_index != -1) {
		_automatic_tab_switch_timer.start(AUTOMATIC_TAB_SWITCH_TIMER_MSEC);
		QApplication::setOverrideCursor(Qt::PointingHandCursor);
	}
	else {
		_automatic_tab_switch_timer.stop();
		QApplication::restoreOverrideCursor();
	}
}


void PVGuiQt::PVWorkspacesTabWidget::switch_tab()
{
	QApplication::restoreOverrideCursor();
	setCurrentIndex(_tab_index);
}

void PVGuiQt::PVWorkspacesTabWidget::tabCloseRequested_Slot(int index)
{
	remove_workspace(index);
}

void PVGuiQt::PVWorkspacesTabWidget::remove_workspace(int index)
{
	assert(index != -1);
	PVGuiQt::PVWorkspace* workspace = qobject_cast<PVGuiQt::PVWorkspace*>(widget(index));
	if (workspace->get_source()) {
		emit workspace_closed(workspace->get_source());
	}

	blockSignals(true);
	removeTab(index);
	blockSignals(false);

	if(count() == 0) {
		emit is_empty();
		hide();
	}
	else {
		setCurrentIndex(index-1);
	}

	workspace->deleteLater();
}
