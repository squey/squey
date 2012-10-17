/**
 * \file PVWorkspacesTabWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <picviz/PVSource.h>

#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVWorkspace.h>

#include <pvkernel/core/lambda_connect.h>

#include <iostream>
#include <QApplication>
#include <QEvent>
#include <QTabBar>
#include <QMouseEvent>
#include <QPushButton>
#include <QDateTime>
#include <QPropertyAnimation>
#include <QLineEdit>

#define AUTOMATIC_TAB_SWITCH_TIMER_MSEC 500
#define TAB_OPENING_EFFECT_MSEC 200

QSize PVGuiQt::PVTabBar::tabSizeHint(int index) const
{
	return QTabBar::tabSizeHint(index);
}

void PVGuiQt::PVTabBar::mouseReleaseEvent(QMouseEvent* event)
{
	// Tabs are closed on middle button click
	if (event->button() == Qt::MidButton) {
		emit tabCloseRequested(tabAt(event->pos()));
	}
	QTabBar::mouseReleaseEvent(event);
}

void PVGuiQt::PVTabBar::mouseDoubleClickEvent(QMouseEvent* event)
{
	int index = tabAt(event->pos());

	if (qobject_cast<PVOpenWorkspace*>(_tab_widget->widget(index))) {
		QLineEdit* line_edit = new QLineEdit(this);
		QRect tab_rect = tabRect(index);
		line_edit->move(tab_rect.topLeft());
		line_edit->resize(QSize(tab_rect.width(), tab_rect.height()));
		line_edit->setText(tabText(index));
		line_edit->show();
		line_edit->setFocus();
		line_edit->setSelection(0, tabText(index).length());

		::connect(line_edit, SIGNAL(editingFinished()), [=] {
			setTabText(index, line_edit->text());
			line_edit->deleteLater();
		});
	}
}

PVGuiQt::PVWorkspacesTabWidget::PVWorkspacesTabWidget(QWidget* parent) :
	QTabWidget(parent),
	_automatic_tab_switch_timer(this)

{
	setObjectName("PVWorkspacesTabWidget");

	_tab_bar = new PVTabBar(this);
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
		addTab(new PVOpenWorkspace(this), "Open workspace", true);
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

int PVGuiQt::PVWorkspacesTabWidget::addTab(PVWorkspaceBase* workspace, const QString & label, bool animation)
{
	setCursor(Qt::ArrowCursor);

	int insert_index = -1;
	if (qobject_cast<PVWorkspace*>(workspace)) {
		insert_index = _workspaces_count++;
	}
	else if (qobject_cast<PVOpenWorkspace*>(workspace)) {
		insert_index = _workspaces_count + _openworkspaces_count++;
	}
	else
	{
		assert(false); // Unknown workspace type
	}

	int index = insertTab(insert_index, workspace, label);
	setCurrentIndex(index);

	if (animation) {
		QPropertyAnimation* animation = new QPropertyAnimation(this, "tab_width");
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
	QString str = QString("QTabBar::tab:selected { width: %1px; color: rgba(0, 0, 0, %2%);}").arg(tab_width).arg((float)tab_width / _tab_width * 100);
	tabBar()->setStyleSheet(tab_width == _tab_width ? "" : str);
}

void PVGuiQt::PVWorkspacesTabWidget::tabInserted(int index)
{
	PVWorkspaceBase* workspace = (PVWorkspaceBase*) widget(index);

	//!\\ Qt is complaining about signal not existing but it definitively does!
	connect(workspace, SIGNAL(try_automatic_tab_switch()), this, SLOT(start_checking_for_automatic_tab_switch()));
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
	PVGuiQt::PVWorkspaceBase* workspace = qobject_cast<PVGuiQt::PVWorkspaceBase*>(widget(index));

	if (PVWorkspace* w = qobject_cast<PVWorkspace*>(workspace)) {
		_workspaces_count--;
		emit workspace_closed(w->get_source());
	}
	else if (qobject_cast<PVOpenWorkspace*>(workspace)) {
		_openworkspaces_count--;
	}
	else
	{
		assert(false); // Unknown workspace type
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
}
