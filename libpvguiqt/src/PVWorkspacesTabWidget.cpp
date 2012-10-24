/**
 * \file PVWorkspacesTabWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <picviz/PVSource.h>

#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVWorkspace.h>

#include <pvkernel/core/lambda_connect.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>

#include <iostream>
#include <QApplication>
#include <QEvent>
#include <QTabBar>
#include <QMouseEvent>
#include <QPushButton>
#include <QDateTime>
#include <QPixmap>
#include <QPainter>
#include <QImage>

#define AUTOMATIC_TAB_SWITCH_TIMER_MSEC 500
#define TAB_OPENING_EFFECT_MSEC 200


bool PVGuiQt::TabRenamerEventFilter::eventFilter(QObject* watched, QEvent* event)
{
	bool rename = false;
	if (event->type() == QEvent::Leave) {
		rename = true;
	}
	else if (event->type() == QEvent::KeyPress) {
		QKeyEvent* key_event = (QKeyEvent*) event;
		rename = key_event->key() == Qt::Key_Return || key_event->key() == Qt::Key_Escape;
	}
	if (rename) {
		_tab_bar->setTabText(_index, _line_edit->text());
		_line_edit->deleteLater();
	}
	return rename;
}

/******************************************************************************
 *
 * PVGuiQt::PVTabBar
 *
 *****************************************************************************/
int PVGuiQt::PVTabBar::count() const
{
	return QTabBar::count() -1;
}

QSize PVGuiQt::PVTabBar::tabSizeHint(int index) const
{
	return QTabBar::tabSizeHint(index);
}

void PVGuiQt::PVTabBar::mouseReleaseEvent(QMouseEvent* event)
{
	_drag_ongoing = false;

	// Tabs are closed on middle button click
	if (event->button() == Qt::MidButton) {
		int tab_index = tabAt(event->pos());
		if (tab_index < count()) {
			emit tabCloseRequested(tab_index);
		}
	}
	QTabBar::mouseReleaseEvent(event);
}

void PVGuiQt::PVTabBar::mouseDoubleClickEvent(QMouseEvent* event)
{
	// Tabs titles are inplace edited on mouse double click
	int index = tabAt(event->pos());

	if (!_tab_widget->_tab_animation_ongoing && index < count() && qobject_cast<PVOpenWorkspace*>(_tab_widget->widget(index))) {
		QLineEdit* line_edit = new QLineEdit(this);
		QRect tab_rect = tabRect(index);
		line_edit->move(tab_rect.topLeft());
		line_edit->resize(QSize(tab_rect.width(), tab_rect.height()));
		line_edit->setText(tabText(index));
		line_edit->show();
		line_edit->setFocus();
		line_edit->setSelection(0, tabText(index).length());
		line_edit->installEventFilter(new TabRenamerEventFilter(this, index, line_edit));
	}
}

void PVGuiQt::PVTabBar::mouseMoveEvent(QMouseEvent* event)
{
	int tab_index = tabAt(event->pos());

	if (tab_index == count()) {
		setCursor(Qt::PointingHandCursor);
	}
	else {
		setCursor(Qt::ArrowCursor);
	}

	bool drag_n_drop = !_drag_ongoing && // No ongoing drag&drop action
					  event->buttons() == Qt::LeftButton && // Drag&drop initialized with left button click
			          tab_index >=0 && tab_index < count() && // Tab is candidate for drag&drop
			          (event->pos() - _drag_start_position).manhattanLength() > QApplication::startDragDistance()*3; // Significant desire to engage drag&drop

	if (drag_n_drop) {
		start_drag(_tab_widget->widget(tab_index));
	}

	QTabBar::mouseMoveEvent(event);
}

void PVGuiQt::PVTabBar::leaveEvent(QEvent* ev)
{
	setCursor(Qt::ArrowCursor);
	QTabBar::leaveEvent(ev);
}

void PVGuiQt::PVTabBar::wheelEvent(QWheelEvent* event)
{
	// Prevent mouse wheel to trigger the creation of a new open workspace
	if (currentIndex() == count()-1 && event->delta() < 0) {
		return;
	}
	QTabBar::wheelEvent(event);
}

void PVGuiQt::PVTabBar::keyPressEvent(QKeyEvent* event)
{
	// Prevent keyboard to trigger the creation of a new open workspace
	if (currentIndex() == count()-1 && event->key() == Qt::Key_Right) {
		return;
	}
	QTabBar::keyPressEvent(event);
}

void PVGuiQt::PVTabBar::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		_drag_start_position = event->pos();
	}

	int index = -1;
	for(int i=0; i < count()+1; i++)
	{
		if(tabRect(i).contains(event->pos())) {
			index = i;
			break;
		}
	}
	if (index == count()) {
		// Special case for "new open workspace" tab
		_tab_widget->addTab(new PVOpenWorkspace(this), "Open workspace");
	}

	QTabBar::mousePressEvent(event);
}

void PVGuiQt::PVTabBar::start_drag(QWidget* workspace)
{
	_drag_ongoing = true;
	QDrag* drag = new QDrag(this);

	QMimeData* mimeData = new QMimeData;

	QByteArray byte_array;
	byte_array.reserve(sizeof(void*));
	byte_array.append((const char*) &workspace, sizeof(void*));

	mimeData->setData("application/x-picviz_workspace", byte_array);

	drag->setMimeData(mimeData);

	// Set semi transparent thumbnail
	QImage opaque = QPixmap::grabWidget(workspace, workspace->rect()).scaledToWidth(200).toImage();
	QPixmap transparent(opaque.size());
	transparent.fill(Qt::transparent);
	QPainter p;
	p.begin(&transparent);
	p.setCompositionMode(QPainter::CompositionMode_Source);
	p.drawPixmap(0, 0, QPixmap::fromImage(opaque));
	p.setCompositionMode(QPainter::CompositionMode_DestinationIn);
	p.fillRect(transparent.rect(), QColor(0, 0, 0, 75));
	p.end();
	drag->setPixmap(transparent);

	QCursor cursor = QCursor(Qt::WaitCursor);
	drag->setDragCursor(cursor.pixmap(), Qt::MoveAction);
	drag->setDragCursor(cursor.pixmap(), Qt::CopyAction);
	drag->setDragCursor(cursor.pixmap(), Qt::IgnoreAction);

	Qt::DropAction action = drag->exec(Qt::CopyAction | Qt::IgnoreAction | Qt::MoveAction | Qt::IgnoreAction);
	if (action == Qt::IgnoreAction) {
		emit _tab_widget->emit_workspace_dragged_outside(workspace);
	}
	if (action == Qt::MoveAction) {
		if (PVWorkspace* w = qobject_cast<PVWorkspace*>(workspace)) {
			_tab_widget->_workspaces_count--;
		}
		else if (qobject_cast<PVOpenWorkspace*>(workspace)) {
			_tab_widget->_openworkspaces_count--;
		}
		else
		{
			assert(false); // Unknown workspace type
		}
	}
	_drag_ongoing = false;
}

/******************************************************************************
 *
 * PVGuiQt::PVWorkspacesTabWidget
 *
 *****************************************************************************/
PVGuiQt::PVWorkspacesTabWidget::PVWorkspacesTabWidget(Picviz::PVScene* scene, QWidget* parent) :
	QTabWidget(parent),
	_scene(scene),
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
	return _tab_bar->count();
}

void PVGuiQt::PVWorkspacesTabWidget::tab_changed(int index)
{
	if (index < count()) {
		Picviz::PVView* view = qobject_cast<PVWorkspaceBase*>(widget(index))->current_view();
		if (view) {
			auto scene_sp = _scene->shared_from_this();
			std::cout << "Picviz::PVScene::select_view: " << view << std::endl;
			PVHive::call<FUNC(Picviz::PVScene::select_view)>(scene_sp, *view);
		}
	}
}

int PVGuiQt::PVWorkspacesTabWidget::addTab(PVWorkspaceBase* workspace, const QString & label)
{
	tabBar()->setCursor(Qt::ArrowCursor);

	// TODO: avoid ugly qobject_cast with virtual methods in PVWorkspaceBase !!
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

	if (true) {
		QPropertyAnimation* animation = new QPropertyAnimation(this, "tab_width");
		animation->setDuration(TAB_OPENING_EFFECT_MSEC);
		animation->setStartValue(25);
		_tab_animated_width = _tab_bar->tabSizeHint(index).width();
		animation->setEndValue(_tab_animated_width);
		animation->start();
	}

	return index;
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

void PVGuiQt::PVWorkspacesTabWidget::tabRemoved(int index)
{
	if(count() == 0) {
		emit is_empty();
		hide();
	}
	else {
		setCurrentIndex(std::min(index, count()-1));
	}
	QTabWidget::tabRemoved(index);
}

void PVGuiQt::PVWorkspacesTabWidget::remove_workspace(int index, bool close_source /*= true*/)
{
	assert(index != -1);
	PVGuiQt::PVWorkspaceBase* workspace = qobject_cast<PVGuiQt::PVWorkspaceBase*>(widget(index));

	if (PVWorkspace* w = qobject_cast<PVWorkspace*>(workspace)) {
		_workspaces_count--;
		if (close_source) {
			_scene->remove_child(*w->get_source());
		}
	}
	else if (qobject_cast<PVOpenWorkspace*>(workspace)) {
		_openworkspaces_count--;
	}
	else
	{
		assert(false); // Unknown workspace type
	}

	if (close_source) {
		QPropertyAnimation* animation = new QPropertyAnimation(this, "tab_width");
		connect(
			animation,
			SIGNAL(stateChanged(QAbstractAnimation::State, QAbstractAnimation::State)),
			this,
			SLOT(animation_state_changed(QAbstractAnimation::State, QAbstractAnimation::State))
		);
		animation->setDuration(TAB_OPENING_EFFECT_MSEC);
		animation->setEndValue(25);
		_tab_animated_width = _tab_bar->tabSizeHint(index).width();
		animation->setStartValue(_tab_animated_width);
		animation->start();

		/*QEventLoop loop;
		loop.connect(this, SIGNAL(animation_finished()), SLOT(quit()));
		loop.exec();*/

		animation->deleteLater();
	}
	else {
		removeTab(currentIndex());
	}
}

void PVGuiQt::PVWorkspacesTabWidget::set_tab_width(int tab_width)
{
	//QString str = QString("QTabBar::tab:selected { width: %1px; color: rgba(0, 0, 0, %2%);}").arg(tab_width).arg((float)tab_width / _tab_animated_width * 100);
	QString str = QString("QTabBar::tab:selected { width: %1px;}").arg(tab_width);
	_tab_animation_ongoing = tab_width != _tab_animated_width;
	tabBar()->setStyleSheet(_tab_animation_ongoing ? str : "");
}

void PVGuiQt::PVWorkspacesTabWidget::animation_state_changed(QAbstractAnimation::State new_state, QAbstractAnimation::State old_state)
{
	if (new_state == QAbstractAnimation::Stopped && old_state == QAbstractAnimation::Running) {
		tabBar()->setStyleSheet("");
		removeTab(currentIndex());
		emit animation_finished();
	}
}
