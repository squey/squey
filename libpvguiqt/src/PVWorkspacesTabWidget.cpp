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
#define TAB_OPENING_EFFECT_MSEC 200

PVGuiQt::PVWorkspacesTabWidget::PVWorkspacesTabWidget(QWidget* parent) :
	QTabWidget(parent),
	_automatic_tab_switch_timer(this)

{
	setObjectName("PVWorkspacesTabWidget");

	_tab_bar = new PVTabBar();
	setTabBar(_tab_bar);

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
		addTab(new PVWorkspace(this), "Open workspace", true);
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

int PVGuiQt::PVWorkspacesTabWidget::addTab(QWidget* page, const QString & label, bool animation)
{
	setCursor(Qt::ArrowCursor);
	int index = insertTab(count(), page, label);
	setCurrentIndex(index);

	if (animation) {
		QPropertyAnimation *animation = new QPropertyAnimation(this, "tab_width");
		animation->setDuration(TAB_OPENING_EFFECT_MSEC);
		animation->setStartValue(25);
		_tab_width = _tab_bar->tabSizeHint(index).width();
		animation->setEndValue(_tab_width);
		animation->start();
	}

	return index;
}

void PVGuiQt::PVWorkspacesTabWidget::set_tab_width(int tab_width)
{
	QString str = QString("QTabBar::tab:selected { width: %1px;}").arg(tab_width);
	tabBar()->setStyleSheet(tab_width == _tab_width ? "" : str);
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
		setCurrentIndex(std::min(index, count()-1));
	}

	workspace->deleteLater();
}
