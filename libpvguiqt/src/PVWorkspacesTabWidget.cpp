/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVWorkspace.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVCallHelper.h>
#include <pvhive/PVObserverSignal.h>

#include <inendi/PVSource.h>

#include <pvkernel/core/lambda_connect.h>

#include <QApplication>
#include <QComboBox>
#include <QDrag>
#include <QEvent>
#include <QImage>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QPropertyAnimation>
#include <QPushButton>

#define AUTOMATIC_TAB_SWITCH_TIMER_MSEC 500
#define TAB_OPENING_EFFECT_MSEC 200

/******************************************************************************
 *
 * PVGuiQt::__impl::TabRenamerEventFilter
 *
 *****************************************************************************/
bool PVGuiQt::__impl::TabRenamerEventFilter::eventFilter(QObject* /*watched*/, QEvent* event)
{
	bool rename = false;
	if (event->type() == QEvent::Leave) {
		rename = true;
	} else if (event->type() == QEvent::KeyPress) {
		QKeyEvent* key_event = (QKeyEvent*)event;
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

PVGuiQt::PVSceneTabBar::PVSceneTabBar(PVWorkspacesTabWidgetBase* tab_widget)
    : _tab_widget(tab_widget)
{
	setTabsClosable(true);
	connect(this, SIGNAL(tabCloseRequested(int)), tab_widget, SLOT(tab_close_requested(int)));
	connect(this, SIGNAL(currentChanged(int)), _tab_widget, SLOT(tab_changed(int)));

	setMovable(true);
	setElideMode(Qt::ElideRight); // Qt::ElideMiddle);
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
	byte_array.append((const char*)&workspace, sizeof(void*));

	mimeData->setData("application/x-inendi_workspace", byte_array);

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

	Qt::DropAction action =
	    drag->exec(Qt::CopyAction | Qt::IgnoreAction | Qt::MoveAction | Qt::IgnoreAction);
	if (action == Qt::IgnoreAction) {
		emit _tab_widget->workspace_dragged_outside(workspace);
	}
	_drag_ongoing = false;
}

void PVGuiQt::PVSceneTabBar::resizeEvent(QResizeEvent* event)
{
	QString stylesheet = "";

	if (count() > 0) {
		int width = _tab_widget->size().width() / count();

		if (width > MIN_WIDTH) {
			QFontMetrics metrics = QFontMetrics(font());

			int i = 0;
			while (i < count() && stylesheet.isEmpty()) {

				if (metrics.width(tabText(i)) > width) {
					stylesheet = QString("QTabBar::tab { max-width: %1px; } ").arg(width);
				}
				i++;
			}
			stylesheet += QString("QTabBar::tab { min-width: %1px; } ").arg(MIN_WIDTH);
		} else
			stylesheet = QString("QTabBar::tab { width: %1px; } ").arg(MIN_WIDTH);
		update();
	}
	_tab_widget->setStyleSheet(stylesheet);

	QTabBar::resizeEvent(event);
}

/******************************************************************************
 *
 * PVGuiQt::PVOpenWorkspaceTabBar
 *
 *****************************************************************************/

PVGuiQt::PVOpenWorkspaceTabBar::PVOpenWorkspaceTabBar(PVOpenWorkspacesTabWidget* tab_widget)
    : PVSceneTabBar(tab_widget)
{
	connect(this, SIGNAL(currentChanged(int)), _tab_widget, SLOT(tab_changed(int)));
}

int PVGuiQt::PVOpenWorkspaceTabBar::count() const
{
	return QTabBar::count() - 1;
}

void PVGuiQt::PVOpenWorkspaceTabBar::wheelEvent(QWheelEvent* event)
{
	// Prevent mouse wheel to trigger the creation of a new open workspace
	if (currentIndex() == count() - 1 && event->delta() < 0) {
		return;
	}
	QTabBar::wheelEvent(event);
}

void PVGuiQt::PVOpenWorkspaceTabBar::keyPressEvent(QKeyEvent* event)
{
	// Prevent keyboard to trigger the creation of a new open workspace
	if (currentIndex() == count() - 1 && event->key() == Qt::Key_Right) {
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
		line_edit->installEventFilter(new __impl::TabRenamerEventFilter(this, index, line_edit));
	}
}

void PVGuiQt::PVOpenWorkspaceTabBar::mousePressEvent(QMouseEvent* event)
{
	int index = -1;
	for (int i = 0; i < count() + 1; i++) {
		if (tabRect(i).contains(event->pos())) {
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
	_tab_widget->add_workspace(open_workspace, QString("Workspace %1").arg(++_workspace_id));

	return open_workspace;
}

/******************************************************************************
 *
 * PVGuiQt::PVWorkspacesTabWidgetBase
 *
 *****************************************************************************/
PVGuiQt::PVWorkspacesTabWidgetBase::PVWorkspacesTabWidgetBase(Inendi::PVRoot& root,
                                                              QWidget* parent /* = 0 */)
    : QTabWidget(parent), _root(root)
{
	setObjectName("PVWorkspacesTabWidget");

	// To get notified of mouse events we must enable mouse tracking on *both*
	// QTabWidget and its underlying QTabBar
	setMouseTracking(true);
	tabBar()->setMouseTracking(true);
}

int PVGuiQt::PVWorkspacesTabWidgetBase::add_workspace(PVWorkspaceBase* workspace,
                                                      const QString& label,
                                                      bool animation /*= true*/)
{
	int index = insertTab(count(), workspace, label);
	setCurrentIndex(index);

	if (animation) {
		QPropertyAnimation* animation = new QPropertyAnimation(this, "tab_width");
		animation->setDuration(TAB_OPENING_EFFECT_MSEC);
		animation->setStartValue(25);
		_tab_animated_width = _tab_bar->tabSizeHint(index).width();
		animation->setEndValue(_tab_animated_width);
		animation->start(QAbstractAnimation::DeleteWhenStopped);
	}

	return index;
}

void PVGuiQt::PVWorkspacesTabWidgetBase::remove_workspace(int index)
{
	QPropertyAnimation* animation = new QPropertyAnimation(this, "tab_width");
	connect(animation, SIGNAL(stateChanged(QAbstractAnimation::State, QAbstractAnimation::State)),
	        this,
	        SLOT(animation_state_changed(QAbstractAnimation::State, QAbstractAnimation::State)));
	blockSignals(true);
	_tab_animation_index = index;
	setCurrentIndex(index); // Force current index in order to get the animation
	// on the selected tab!
	blockSignals(false);
	_tab_animation_index = index;
	animation->setDuration(TAB_OPENING_EFFECT_MSEC);
	animation->setEndValue(25);
	_tab_animated_width = _tab_bar->tabSizeHint(index).width();
	animation->setStartValue(_tab_animated_width);
	animation->start(QAbstractAnimation::DeleteWhenStopped);
}

void PVGuiQt::PVWorkspacesTabWidgetBase::set_tab_width(int tab_width)
{
	// QString str = QString("QTabBar::tab:selected { width: %1px; color: rgba(0,
	// 0, 0, %2%);}").arg(tab_width).arg((float)tab_width / _tab_animated_width *
	// 100);
	QString str = QString("QTabBar::tab:selected { width: %1px;}").arg(tab_width);
	_tab_animation_ongoing = tab_width != _tab_animated_width;
	tabBar()->setStyleSheet(_tab_animation_ongoing ? str : "");
}

void PVGuiQt::PVWorkspacesTabWidgetBase::animation_state_changed(
    QAbstractAnimation::State new_state, QAbstractAnimation::State old_state)
{
	if (new_state == QAbstractAnimation::Stopped && old_state == QAbstractAnimation::Running) {
		tabBar()->setStyleSheet("");
		widget(_tab_animation_index)->deleteLater();
		removeTab(_tab_animation_index);
		sender()->deleteLater();
	}
}

void PVGuiQt::PVWorkspacesTabWidgetBase::tab_close_requested(int index)
{
	PVWorkspaceBase* workspace = qobject_cast<PVWorkspaceBase*>(widget(index));
	workspace->displays_about_to_be_deleted();
	remove_workspace(index);
}

void PVGuiQt::PVWorkspacesTabWidgetBase::resizeEvent(QResizeEvent* event)
{
	_tab_bar->resizeEvent(event);
	QTabWidget::resizeEvent(event);
}

/******************************************************************************
 *
 * PVGuiQt::PVSceneWorkspacesTabWidget
 *
 *****************************************************************************/
PVGuiQt::PVSceneWorkspacesTabWidget::PVSceneWorkspacesTabWidget(Inendi::PVScene& scene,
                                                                QWidget* parent /* = 0 */)
    : PVWorkspacesTabWidgetBase(*scene.get_parent<Inendi::PVRoot>(), parent)
{
	Inendi::PVScene_sp scene_p = scene.shared_from_this();
	PVHive::get().register_observer(scene_p, _obs_scene);
	_obs_scene.connect_refresh(this, SLOT(set_project_modified()));
	_obs_scene.set_accept_recursive_refreshes(true);

	_tab_bar = new PVSceneTabBar(this);
	setTabBar(_tab_bar);
}

void PVGuiQt::PVSceneWorkspacesTabWidget::set_project_modified(bool modified /* = true */,
                                                               QString path /*= QString()*/)
{
	if (!_project_modified && modified) {
		emit project_modified(true);
	} else if (_project_modified && !modified) {
		_project_untitled = false;
		emit project_modified(false, path);
	}
	_project_modified = modified;
}

void PVGuiQt::PVSceneWorkspacesTabWidget::tabRemoved(int index)
{
	if (count() == 0) {
		emit is_empty();
		hide();
	} else {
		setCurrentIndex(std::min(index, count() - 1));
	}
	QTabWidget::tabRemoved(index);
}

void PVGuiQt::PVSceneWorkspacesTabWidget::tab_changed(int index)
{
	if (index == -1)
		return;

	PVSourceWorkspace* workspace = qobject_cast<PVSourceWorkspace*>(widget(index));
	assert(workspace);
	Inendi::PVRoot_sp root_sp = get_scene()->get_parent<Inendi::PVRoot>()->shared_from_this();
	PVHive::call<FUNC(Inendi::PVRoot::select_source)>(root_sp, *workspace->get_source());
}

/******************************************************************************
 *
 * PVGuiQt::PVOpenWorkspacesTabWidget
 *
 *****************************************************************************/
PVGuiQt::PVOpenWorkspacesTabWidget::PVOpenWorkspacesTabWidget(Inendi::PVRoot& root,
                                                              QWidget* parent /* = 0 */)
    : PVWorkspacesTabWidgetBase(root, parent), _automatic_tab_switch_timer(this)
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

	((PVOpenWorkspaceTabBar*)_tab_bar)->create_new_workspace();
}

void PVGuiQt::PVOpenWorkspacesTabWidget::tabInserted(int index)
{
	if (count() > 0) {
		PVWorkspaceBase* workspace = (PVWorkspaceBase*)widget(index);
		connect(workspace, SIGNAL(try_automatic_tab_switch()), this,
		        SLOT(start_checking_for_automatic_tab_switch()));
	}
}

void PVGuiQt::PVOpenWorkspacesTabWidget::tabRemoved(int index)
{
	if (index == count()) {
		// Prevent selection of open workspace "+" tab.
		setCurrentIndex(index - 1);
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
		// QApplication::setOverrideCursor(Qt::PointingHandCursor);
	} else {
		_automatic_tab_switch_timer.stop();
		// QApplication::restoreOverrideCursor();
	}
}

void PVGuiQt::PVOpenWorkspacesTabWidget::switch_tab()
{
	// QApplication::restoreOverrideCursor();
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
