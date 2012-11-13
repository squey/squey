/**
 * \file PVWorkspacesTabWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <picviz/PVSource.h>

#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVWorkspace.h>

#include <pvkernel/core/lambda_connect.h>

#include <picviz/PVAD2GView.h>

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
	//setCursor(Qt::ArrowCursor);
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

PVGuiQt::PVOpenWorkspaceTabBar::PVOpenWorkspaceTabBar(PVOpenWorkspacesTabWidget* tab_widget) : PVSceneTabBar(tab_widget)
{
	connect(this, SIGNAL(currentChanged(int)), _tab_widget, SLOT(tab_changed(int)));
}

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

PVGuiQt::PVOpenWorkspace* PVGuiQt::PVOpenWorkspaceTabBar::create_new_workspace()
{
	PVOpenWorkspace* open_workspace = new PVOpenWorkspace(this);
	_tab_widget->addTab(open_workspace, QString("Workspace %1").arg(++_workspace_id));

	return open_workspace;
}

/******************************************************************************
 *
 * PVGuiQt::PVWorkspacesTabWidgetBase
 *
 *****************************************************************************/
PVGuiQt::PVWorkspacesTabWidgetBase::PVWorkspacesTabWidgetBase(Picviz::PVRoot& root, QWidget* parent /* = 0 */) :
	QTabWidget(parent),
	_root(root)
{
	setObjectName("PVWorkspacesTabWidget");

	// To get notified of mouse events we must enable mouse tracking on *both* QTabWidget and its underlying QTabBar
	setMouseTracking(true);
	tabBar()->setMouseTracking(true);

	_combo_box = new QComboBox();
	connect(_combo_box, SIGNAL(activated(int)), this, SLOT(correlation_changed(int)));
	setCornerWidget(_combo_box, Qt::TopRightCorner);

	// Register observers for correlations
	Picviz::PVRoot_sp root_sp = root.shared_from_this();
	_obs = new PVHive::PVObserverSignal<Picviz::PVRoot>(this);
	PVHive::get().register_observer(root_sp, [=](Picviz::PVRoot& root) { return &root.get_correlations(); }, *_obs);
	_obs->connect_refresh(this, SLOT(update_correlations_list()));
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
		blockSignals(true);
		_tab_animation_index = index;
		setCurrentIndex(index); // Force current index in order to get the animation on the selected tab!
		blockSignals(false);
		_tab_animation_index = index;
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
		removeTab(_tab_animation_index);
		sender()->deleteLater();
		emit animation_finished();
	}
}

void PVGuiQt::PVWorkspacesTabWidgetBase::correlation_changed(int index)
{
	get_root().select_correlation(get_correlation());
}

void PVGuiQt::PVWorkspacesTabWidgetBase::update_correlations_list()
{
	_combo_box->clear();
	_combo_box->addItem("(No correlation)");
	Picviz::PVRoot::correlations_t corrs = get_correlations();
	for (Picviz::PVAD2GView_p const& c: corrs) {
		_combo_box->addItem(c->get_name());
		_combo_box->setItemData(_combo_box->count()-1, qVariantFromValue((void*) c.get()), Qt::UserRole);
	}
	Picviz::PVAD2GView* correlation = get_correlation();
	get_root().select_correlation(correlation);
	_combo_box->setCurrentIndex(get_index_from_correlation(correlation));
}

int PVGuiQt::PVWorkspacesTabWidgetBase::get_index_from_correlation(void* correlation)
{
	for (int i = 0; i < _combo_box->count(); i++) {
		void* corr = _combo_box->itemData(i, Qt::UserRole).value<void*>();
		if (correlation == corr) {
			return i;
		}
	}
	return 0;
}

void PVGuiQt::PVWorkspacesTabWidgetBase::tabCloseRequested_Slot(int index)
{
	PVWorkspaceBase* workspace = qobject_cast<PVWorkspaceBase*>(widget(index));
	workspace->displays_about_to_be_deleted();
	remove_workspace(index);
}

QList<PVGuiQt::PVWorkspaceBase*> PVGuiQt::PVWorkspacesTabWidgetBase::list_workspaces() const
{
	QList<PVWorkspaceBase*> ret;
	for (int i = 0; i < count(); i++) {
		PVWorkspaceBase* workspace = qobject_cast<PVWorkspaceBase*>(widget(i));
		assert(workspace);
		ret << workspace;
	}
	return ret;
}

/******************************************************************************
 *
 * PVGuiQt::PVSceneWorkspacesTabWidget
 *
 *****************************************************************************/
PVGuiQt::PVSceneWorkspacesTabWidget::PVSceneWorkspacesTabWidget(Picviz::PVScene& scene, QWidget* parent /* = 0 */) :
	PVWorkspacesTabWidgetBase(*scene.get_parent<Picviz::PVRoot>(), parent),
	_save_scene_func_observer(this)
{
	Picviz::PVScene_sp scene_p = scene.shared_from_this();
	PVHive::get().register_observer(scene_p, _obs_scene);
	_obs_scene.connect_refresh(this, SLOT(set_project_modified()));

	// AG: we need to clear the way GUI-objects related to data-tree ones are created and destroyed.
	// This way is one of the good ones, that is keeping track thanks to the hive of what exists in the data-tree and
	// react in such consequence.
	//_obs_scene.connect_refresh(this, SLOT(check_new_sources()));

	PVHive::get().register_func_observer(scene_p, _save_scene_func_observer);
	_obs_scene.set_accept_recursive_refreshes(true);

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
	_correlation = (Picviz::PVAD2GView*) _combo_box->itemData(index, Qt::UserRole).value<void*>();

	PVWorkspacesTabWidgetBase::correlation_changed(index);
}

void PVGuiQt::PVSceneWorkspacesTabWidget::remove_workspace(int index, bool close_source /*= true*/)
{
	assert(index != -1);
	PVGuiQt::PVWorkspace* workspace = qobject_cast<PVGuiQt::PVWorkspace*>(widget(index));

	if (workspace && close_source) {
		get_scene()->remove_child(*workspace->get_source());
	}

	PVWorkspacesTabWidgetBase::remove_workspace(index, close_source);
}

void PVGuiQt::PVSceneWorkspacesTabWidget::tab_changed(int index)
{
	if (index == -1) return;

	PVWorkspace* workspace = qobject_cast<PVWorkspace*>(widget(index));
	assert(workspace);
	Picviz::PVRoot_sp root_sp = get_scene()->get_parent<Picviz::PVRoot>()->shared_from_this();
	PVHive::call<FUNC(Picviz::PVRoot::select_source)>(root_sp, *workspace->get_source());
}

void PVGuiQt::PVSceneWorkspacesTabWidget::check_new_sources()
{
	QList<Picviz::PVSource*> known_srcs = list_sources();
	for (Picviz::PVSource_sp& src: get_scene()->get_children<Picviz::PVSource>()) {
		if (known_srcs.contains(src.get())) {
			continue;
		}

		PVWorkspace* new_workspace = new PVWorkspace(src.get());
		addTab(new_workspace, src->get_name());
	}
}

QList<Picviz::PVSource*> PVGuiQt::PVSceneWorkspacesTabWidget::list_sources() const
{
	QList<Picviz::PVSource*> ret;
	for (PVWorkspaceBase* w: list_workspaces()) {
		ret << qobject_cast<PVWorkspace*>(w)->get_source();
	}
	return ret;
}

/******************************************************************************
 *
 * PVGuiQt::PVOpenWorkspacesTabWidget
 *
 *****************************************************************************/
PVGuiQt::PVOpenWorkspacesTabWidget::PVOpenWorkspacesTabWidget(Picviz::PVRoot& root,QWidget* parent /* = 0 */) :
	PVWorkspacesTabWidgetBase(root, parent),
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
		connect(workspace, SIGNAL(try_automatic_tab_switch()), this, SLOT(start_checking_for_automatic_tab_switch()));
	}
}

void PVGuiQt::PVOpenWorkspacesTabWidget::tab_changed(int index)
{
	if (index == count()) {
		/*int idx = std::max(0, count()-1);
		std::cout << "setCurrentIndex(idx)=" << idx << std::endl;
		setCurrentIndex(idx);*/
	}
	else {
		if (_combo_box) {
			PVOpenWorkspace * open_workspace = (PVOpenWorkspace*) widget(index);
			if (open_workspace) {
				Picviz::PVAD2GView* correlation = open_workspace->get_correlation();
				_combo_box->setCurrentIndex(get_index_from_correlation(correlation));
				get_root().select_correlation(correlation);
			}
		}
	}
}

Picviz::PVAD2GView* PVGuiQt::PVOpenWorkspacesTabWidget::get_correlation()
{
	PVGuiQt::PVOpenWorkspace* open_workspace = qobject_cast<PVGuiQt::PVOpenWorkspace*>(currentWidget());
	if (open_workspace) {
		return open_workspace->get_correlation();
	}
	return nullptr;
}

void PVGuiQt::PVOpenWorkspacesTabWidget::correlation_changed(int index)
{
	PVGuiQt::PVOpenWorkspace* open_workspace = qobject_cast<PVGuiQt::PVOpenWorkspace*>(currentWidget());
	if (open_workspace) {
		Picviz::PVAD2GView* correlation = (Picviz::PVAD2GView*) _combo_box->itemData(index, Qt::UserRole).value<void*>();
		open_workspace->set_correlation(correlation);
	}

	PVWorkspacesTabWidgetBase::correlation_changed(index);
}

void PVGuiQt::PVOpenWorkspacesTabWidget::tabRemoved(int index)
{
	if (count() == 0) {
		// If there isn't any open workspace anymore, disable correlation.
		_combo_box->setCurrentIndex(0);
	}
	else if (index == count()) {
		// Prevent selection of open workspace "+" tab.
		setCurrentIndex(index-1);
	}
}

PVGuiQt::PVOpenWorkspace* PVGuiQt::PVOpenWorkspacesTabWidget::current_workspace() const
{
	return qobject_cast<PVOpenWorkspace*>(currentWidget());
}

void PVGuiQt::PVOpenWorkspacesTabWidget::start_checking_for_automatic_tab_switch()
{
	QPoint mouse_pos = tabBar()->mapFromGlobal(QCursor::pos());
	_tab_switch_index = tabBar()->tabAt(mouse_pos);

	if (_tab_switch_index != -1) {
		_automatic_tab_switch_timer.start(AUTOMATIC_TAB_SWITCH_TIMER_MSEC);
		//QApplication::setOverrideCursor(Qt::PointingHandCursor);
	}
	else {
		_automatic_tab_switch_timer.stop();
		//QApplication::restoreOverrideCursor();
	}
}

void PVGuiQt::PVOpenWorkspacesTabWidget::switch_tab()
{
	//QApplication::restoreOverrideCursor();
	setCurrentIndex(_tab_switch_index);
}

PVGuiQt::PVOpenWorkspace* PVGuiQt::PVOpenWorkspacesTabWidget::current_workspace_or_create()
{
	PVOpenWorkspace* ret = current_workspace();
	if (!ret) {
		ret = _tab_bar->create_new_workspace();
	}
	return ret;
}
