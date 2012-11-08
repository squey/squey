/**
 * \file PVWorkspacesTabWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <picviz/PVSource.h>

#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVWorkspace.h>

#include <pvkernel/core/lambda_connect.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVCallHelper.h>
#include <pvhive/PVObserverSignal.h>

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

void PVGuiQt::__impl::PVSaveSceneToFileFuncObserver::update(const arguments_deep_copy_type& args) const
{
	_parent->set_project_modified(false, std::get<0>(args));
}

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
 * PVGuiQt::PVSceneTabBar
 *
 *****************************************************************************/

PVGuiQt::PVSceneTabBar::PVSceneTabBar(PVWorkspacesTabWidgetBase* tab_widget) : _tab_widget(tab_widget)
{
	setTabsClosable(true);
	connect(this, SIGNAL(tabCloseRequested(int)), tab_widget, SLOT(tabCloseRequested_Slot(int)));
	connect(this, SIGNAL(currentChanged(int)), _tab_widget, SLOT(tab_changed(int)));
}

int PVGuiQt::PVSceneTabBar::count() const
{
	return QTabBar::count();
}

QSize PVGuiQt::PVSceneTabBar::tabSizeHint(int index) const
{
	return QTabBar::tabSizeHint(index);
}

void PVGuiQt::PVSceneTabBar::mouseReleaseEvent(QMouseEvent* event)
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

void PVGuiQt::PVSceneTabBar::mouseMoveEvent(QMouseEvent* event)
{
	// Drag&drop is disabled for the moment...
	/*int tab_index = tabAt(event->pos());

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

	QTabBar::mouseMoveEvent(event);*/
}

void PVGuiQt::PVSceneTabBar::leaveEvent(QEvent* ev)
{
	setCursor(Qt::ArrowCursor);
	QTabBar::leaveEvent(ev);
}

void PVGuiQt::PVSceneTabBar::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		_drag_start_position = event->pos();
	}
	QTabBar::mousePressEvent(event);
}

void PVGuiQt::PVSceneTabBar::start_drag(QWidget* workspace)
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
	_drag_ongoing = false;
}

/******************************************************************************
 *
 * PVGuiQt::PVOpenWorkspaceTabBar
 *
 *****************************************************************************/

int PVGuiQt::PVOpenWorkspaceTabBar::count() const
{
	return QTabBar::count() -1;
}

void PVGuiQt::PVOpenWorkspaceTabBar::wheelEvent(QWheelEvent* event)
{
	// Prevent mouse wheel to trigger the creation of a new open workspace
	if (currentIndex() == count()-1 && event->delta() < 0) {
		return;
	}
	QTabBar::wheelEvent(event);
}

void PVGuiQt::PVOpenWorkspaceTabBar::keyPressEvent(QKeyEvent* event)
{
	// Prevent keyboard to trigger the creation of a new open workspace
	if (currentIndex() == count()-1 && event->key() == Qt::Key_Right) {
		return;
	}
	QTabBar::keyPressEvent(event);
}

void PVGuiQt::PVOpenWorkspaceTabBar::mouseDoubleClickEvent(QMouseEvent* event)
{
	// Tabs titles are inplace edited on mouse double click
	int index = tabAt(event->pos());
	if (!_tab_widget->_tab_animation_ongoing && index < count()) {
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

void PVGuiQt::PVOpenWorkspaceTabBar::mousePressEvent(QMouseEvent* event)
{
	int index = -1;
	for(int i=0; i < count()+1; i++) {
		if(tabRect(i).contains(event->pos())) {
			index = i;
			break;
		}
	}
	if (index == count()) {
		// Special case for "new open workspace" tab
		create_new_workspace();
	}

	PVSceneTabBar::mousePressEvent(event);
}

void PVGuiQt::PVOpenWorkspaceTabBar::create_new_workspace()
{
	_tab_widget->addTab(new PVOpenWorkspace(this), QString("Workspace %1").arg(_tab_widget->count()+1));
}

/******************************************************************************
 *
 * PVGuiQt::PVWorkspacesTabWidgetBase
 *
 *****************************************************************************/
PVGuiQt::PVWorkspacesTabWidgetBase::PVWorkspacesTabWidgetBase(QWidget* parent /* = 0 */) :
	QTabWidget(parent)
{
	setObjectName("PVWorkspacesTabWidget");

	// To get notified of mouse events we must enable mouse tracking on *both* QTabWidget and its underlying QTabBar
	setMouseTracking(true);
	tabBar()->setMouseTracking(true);

	_combo_box = new QComboBox();
	connect(_combo_box, SIGNAL(activated(int)), this, SLOT(correlation_changed(int)));
	setCornerWidget(_combo_box, Qt::TopRightCorner);

	// Register observers for correlations
	Picviz::PVRoot_sp root_sp = Picviz::PVRoot::get_root_sp();
	PVHive::PVObserverSignal<Picviz::PVRoot>* obs = new PVHive::PVObserverSignal<Picviz::PVRoot>(this);
	PVHive::get().register_observer(root_sp, [=](Picviz::PVRoot& root) { return &root.get_correlations(); }, *obs);
	obs->connect_refresh(this, SLOT(update_correlations_list()));
}


int PVGuiQt::PVWorkspacesTabWidgetBase::addTab(PVWorkspaceBase* workspace, const QString & label)
{
	int index = insertTab(count(), workspace, label);
	setCurrentIndex(index);

	QPropertyAnimation* animation = new QPropertyAnimation(this, "tab_width");
	animation->setDuration(TAB_OPENING_EFFECT_MSEC);
	animation->setStartValue(25);
	_tab_animated_width = _tab_bar->tabSizeHint(index).width();
	animation->setEndValue(_tab_animated_width);
	animation->start();

	return index;
}

void PVGuiQt::PVWorkspacesTabWidgetBase::set_tab_width(int tab_width)
{
	//QString str = QString("QTabBar::tab:selected { width: %1px; color: rgba(0, 0, 0, %2%);}").arg(tab_width).arg((float)tab_width / _tab_animated_width * 100);
	QString str = QString("QTabBar::tab:selected { width: %1px;}").arg(tab_width);
	_tab_animation_ongoing = tab_width != _tab_animated_width;
	tabBar()->setStyleSheet(_tab_animation_ongoing ? str : "");
}

void PVGuiQt::PVWorkspacesTabWidgetBase::remove_workspace(int index, bool close_animation /*= true*/)
{
	if (close_animation) {
		QPropertyAnimation* animation = new QPropertyAnimation(this, "tab_width");
		connect(
			animation,
			SIGNAL(stateChanged(QAbstractAnimation::State, QAbstractAnimation::State)),
			this,
			SLOT(animation_state_changed(QAbstractAnimation::State, QAbstractAnimation::State))
		);
		setCurrentIndex(index);
		animation->setDuration(TAB_OPENING_EFFECT_MSEC);
		animation->setEndValue(25);
		_tab_animated_width = _tab_bar->tabSizeHint(index).width();
		animation->setStartValue(_tab_animated_width);
		animation->start();

		/*QEventLoop loop;
		loop.connect(this, SIGNAL(animation_finished()), SLOT(quit()));
		loop.exec();*/
	}
	else {
		removeTab(index);
	}
}

void PVGuiQt::PVWorkspacesTabWidgetBase::animation_state_changed(QAbstractAnimation::State new_state, QAbstractAnimation::State old_state)
{
	if (new_state == QAbstractAnimation::Stopped && old_state == QAbstractAnimation::Running) {
		tabBar()->setStyleSheet("");
		removeTab(currentIndex());
		sender()->deleteLater();
		emit animation_finished();
	}
}

void PVGuiQt::PVWorkspacesTabWidgetBase::correlation_changed(int index)
{
	Picviz::PVRoot::get_root().select_correlation(index-1);
}

void PVGuiQt::PVWorkspacesTabWidgetBase::update_correlations_list()
{
	_combo_box->clear();
	_combo_box->addItem("(No correlation)");
	for (auto correlation : Picviz::PVRoot::get_root().get_correlations()) {
		_combo_box->addItem(correlation->get_name());
	}
	int index = get_correlation_index();
	if (index == -1) {
		Picviz::PVRoot::get_root().select_correlation(index++);
	}
	_combo_box->setCurrentIndex(index);
}

void PVGuiQt::PVWorkspacesTabWidgetBase::tabCloseRequested_Slot(int index)
{
	remove_workspace(index);
}
/******************************************************************************
 *
 * PVGuiQt::PVSceneWorkspacesTabWidget
 *
 *****************************************************************************/
PVGuiQt::PVSceneWorkspacesTabWidget::PVSceneWorkspacesTabWidget(Picviz::PVScene_p scene_p, QWidget* parent /* = 0 */) :
	PVWorkspacesTabWidgetBase(parent),
	_scene_p(scene_p),
	_save_scene_func_observer(this)
{
	PVHive::get().register_observer(scene_p, _obs_scene);
	_obs_scene.connect_refresh(this, SLOT(set_project_modified()));

	PVHive::get().register_func_observer(scene_p, _save_scene_func_observer);

	_tab_bar = new PVSceneTabBar(this);
	setTabBar(_tab_bar);

	update_correlations_list();
}

void PVGuiQt::PVSceneWorkspacesTabWidget::set_project_modified(bool modified /* = true */, QString path /*= QString()*/)
{
	if (!_project_modified && modified) {
		emit project_modified(true);
	}
	else if (_project_modified && !modified) {
		_project_untitled = false;
		emit project_modified(false, path);

	}
	_project_modified = modified;
}

void PVGuiQt::PVSceneWorkspacesTabWidget::tabRemoved(int index)
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

void PVGuiQt::PVSceneWorkspacesTabWidget::correlation_changed(int index)
{
	PVWorkspacesTabWidgetBase::correlation_changed(index);
	_correlation_name = _combo_box->itemText(index);
}

void PVGuiQt::PVSceneWorkspacesTabWidget::remove_workspace(int index, bool close_source /*= true*/)
{
	assert(index != -1);
	PVGuiQt::PVWorkspace* workspace = qobject_cast<PVGuiQt::PVWorkspace*>(widget(index));

	if (workspace && close_source) {
		_scene_p->remove_child(*workspace->get_source());
	}

	PVWorkspacesTabWidgetBase::remove_workspace(index, close_source);
}

void PVGuiQt::PVSceneWorkspacesTabWidget::tab_changed(int index)
{
	if (index == -1) return;

	Picviz::PVView* view = qobject_cast<PVWorkspaceBase*>(widget(index))->current_view();
	if (view) {
		PVHive::call<FUNC(Picviz::PVScene::select_view)>(_scene_p, *view);
	}
}
/******************************************************************************
 *
 * PVGuiQt::PVOpenWorkspacesTabWidget
 *
 *****************************************************************************/
PVGuiQt::PVOpenWorkspacesTabWidget::PVOpenWorkspacesTabWidget(QWidget* parent /* = 0 */) :
	PVWorkspacesTabWidgetBase(parent),
	_automatic_tab_switch_timer(this)
{
	_tab_bar = new PVOpenWorkspaceTabBar(this);
	setTabBar(_tab_bar);

	QWidget* new_tab = new QWidget();
	QTabWidget::addTab(new_tab, QIcon(":/more.png"), "");
	setTabToolTip(0, tr("New workspace"));
	QPushButton* hidden_close_button = new QPushButton();
	hidden_close_button->resize(QSize(0, 0));
	tabBar()->setTabButton(0, QTabBar::RightSide, hidden_close_button);

	// Automatic tab switching handling  for drag&drop
	_automatic_tab_switch_timer.setSingleShot(true);
	connect(&_automatic_tab_switch_timer, SIGNAL(timeout()), this, SLOT(switch_tab()));

	((PVOpenWorkspaceTabBar*) _tab_bar)->create_new_workspace();

	update_correlations_list();
}

void PVGuiQt::PVOpenWorkspacesTabWidget::tabInserted(int index)
{
	if (count() > 0) {
		PVWorkspaceBase* workspace = (PVWorkspaceBase*) widget(index);
		std::cout << "tabInserted: workspace=" << workspace << std::endl;
		connect(workspace, SIGNAL(try_automatic_tab_switch()), this, SLOT(start_checking_for_automatic_tab_switch()));
	}
}

void PVGuiQt::PVOpenWorkspacesTabWidget::tab_changed(int index)
{

	PVWorkspaceBase* workspace = (PVWorkspaceBase*) widget(index);
	std::cout << "workspace=" << workspace << std::endl;
	if (index == count()) {
		setCurrentIndex(count()-1);
	}
	else {
		if (_combo_box) {
			PVOpenWorkspace * open_workspace = (PVOpenWorkspace*) widget(index);
			if (open_workspace) {
				_combo_box->setCurrentIndex(open_workspace->get_correlation_index());
			}
		}
	}
}

int PVGuiQt::PVOpenWorkspacesTabWidget::get_correlation_index()
{
	PVGuiQt::PVOpenWorkspace* open_workspace = qobject_cast<PVGuiQt::PVOpenWorkspace*>(currentWidget());
	if (open_workspace) {
		return open_workspace->get_correlation_index();
	}
	return 0;
}

void PVGuiQt::PVOpenWorkspacesTabWidget::correlation_changed(int index)
{
	PVWorkspacesTabWidgetBase::correlation_changed(index);
	PVGuiQt::PVOpenWorkspace* open_workspace = qobject_cast<PVGuiQt::PVOpenWorkspace*>(currentWidget());
	if (open_workspace) {
		open_workspace->set_correlation_index(index);
	}
}

void PVGuiQt::PVOpenWorkspacesTabWidget::tabRemoved(int index)
{
	if (count() == 0) {
		_combo_box->setCurrentIndex(0);
	}
	QTabWidget::tabRemoved(index);
}

void PVGuiQt::PVOpenWorkspacesTabWidget::start_checking_for_automatic_tab_switch()
{
	std::cout << "start_checking_for_automatic_tab_switch" << std::endl;
	QPoint mouse_pos = tabBar()->mapFromGlobal(QCursor::pos());
	_tab_switch_index = tabBar()->tabAt(mouse_pos);

	if (_tab_switch_index != -1) {
		_automatic_tab_switch_timer.start(AUTOMATIC_TAB_SWITCH_TIMER_MSEC);
		QApplication::setOverrideCursor(Qt::PointingHandCursor);
	}
	else {
		_automatic_tab_switch_timer.stop();
		QApplication::restoreOverrideCursor();
	}
}

void PVGuiQt::PVOpenWorkspacesTabWidget::switch_tab()
{
	QApplication::restoreOverrideCursor();
	setCurrentIndex(_tab_switch_index);
}
