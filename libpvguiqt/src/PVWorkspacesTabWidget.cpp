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

#include <QDrag>
#include <QImage>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QPropertyAnimation>

#define TAB_OPENING_EFFECT_MSEC 200

/******************************************************************************
 *
 * PVGuiQt::PVSceneTabBar
 *
 *****************************************************************************/

PVGuiQt::PVSceneTabBar::PVSceneTabBar(PVSceneWorkspacesTabWidget* tab_widget)
    : _tab_widget(tab_widget)
{
	setTabsClosable(true);
	connect(this, SIGNAL(tabCloseRequested(int)), tab_widget, SLOT(tab_close_requested(int)));
	connect(this, SIGNAL(currentChanged(int)), _tab_widget, SLOT(tab_changed(int)));

	setMovable(true);
	setElideMode(Qt::ElideRight);
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

		// FIXME(pbrunet) : setting min_width to MIN_WIDTH and max_width to max(width, MIN_WIDTH)
		// should do the same
		if (width > MIN_WIDTH) {
			QFontMetrics metrics = QFontMetrics(font());
			for (int i = 0; i < count(); i++) {
				if (metrics.width(tabText(i)) > width) {
					stylesheet = QString("QTabBar::tab { max-width: %1px; } ").arg(width);
					break;
				}
			}
			stylesheet += QString("QTabBar::tab { min-width: %1px; } ").arg(MIN_WIDTH);
		} else {
			stylesheet = QString("QTabBar::tab { width: %1px; } ").arg(MIN_WIDTH);
		}
		update();
	}
	_tab_widget->setStyleSheet(stylesheet);

	QTabBar::resizeEvent(event);
}

/******************************************************************************
 *
 * PVGuiQt::PVSceneWorkspacesTabWidget
 *
 *****************************************************************************/
PVGuiQt::PVSceneWorkspacesTabWidget::PVSceneWorkspacesTabWidget(Inendi::PVScene& scene,
                                                                QWidget* parent /* = 0 */)
    : QTabWidget(parent)
{
	setObjectName("PVWorkspacesTabWidget");

	// To get notified of mouse events we must enable mouse tracking on *both*
	// QTabWidget and its underlying QTabBar
	setMouseTracking(true);
	tabBar()->setMouseTracking(true);

	Inendi::PVScene_sp scene_p = scene.shared_from_this();
	PVHive::get().register_observer(scene_p, _obs_scene);
	_obs_scene.connect_refresh(this, SLOT(set_project_modified()));
	_obs_scene.set_accept_recursive_refreshes(true);

	setTabBar(new PVSceneTabBar(this));
}

void PVGuiQt::PVSceneWorkspacesTabWidget::add_workspace(PVWorkspaceBase* workspace,
                                                        const QString& label)
{
	// Add the new workspace and select it
	int index = addTab(workspace, label);
	setCurrentIndex(index);

	// Add an animation on the tabBar.
	QPropertyAnimation* animation = new QPropertyAnimation(this, "tab_width");
	animation->setDuration(TAB_OPENING_EFFECT_MSEC);
	animation->setStartValue(25);
	animation->setEndValue(tabBar()->tabRect(index).width());
	animation->start(QAbstractAnimation::DeleteWhenStopped);

	connect(animation, &QPropertyAnimation::finished, this,
	        &PVSceneWorkspacesTabWidget::animation_finished);
}

void PVGuiQt::PVSceneWorkspacesTabWidget::remove_workspace(int index)
{
	QPropertyAnimation* animation = new QPropertyAnimation(this, "tab_width");
	blockSignals(true);
	setCurrentIndex(index); // Force current index in order to get the animation
	// on the selected tab!
	blockSignals(false);
	animation->setDuration(TAB_OPENING_EFFECT_MSEC);
	animation->setStartValue(tabBar()->tabRect(index).width());
	animation->setEndValue(25);
	animation->start(QAbstractAnimation::DeleteWhenStopped);

	connect(animation, &QPropertyAnimation::finished, [this, index]() {
		tabBar()->setStyleSheet("");
		QWidget* w = widget(index);
		removeTab(index);
		if (count() == 0) {
			delete w;
			emit is_empty();
			hide();
		} else {
			get_scene().remove_child(*qobject_cast<PVGuiQt::PVSourceWorkspace*>(w)->get_source());
			delete w;
			setCurrentIndex(std::min(index, count() - 1));
		}
	});
}

void PVGuiQt::PVSceneWorkspacesTabWidget::set_tab_width(int tab_width)
{
	tabBar()->setStyleSheet(QString("QTabBar::tab:selected { width: %1px;}").arg(tab_width));
}

void PVGuiQt::PVSceneWorkspacesTabWidget::animation_finished()
{
	tabBar()->setStyleSheet("");
}

void PVGuiQt::PVSceneWorkspacesTabWidget::tab_close_requested(int index)
{
	PVWorkspaceBase* workspace = qobject_cast<PVWorkspaceBase*>(widget(index));
	workspace->displays_about_to_be_deleted();
	remove_workspace(index);
}

void PVGuiQt::PVSceneWorkspacesTabWidget::resizeEvent(QResizeEvent* event)
{
	// FIXME(pbrunet) : Check if it is not done by default.
	dynamic_cast<PVSceneTabBar*>(tabBar())->resizeEvent(event);
	QTabWidget::resizeEvent(event);
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

void PVGuiQt::PVSceneWorkspacesTabWidget::tab_changed(int index)
{
	if (index == -1) {
		return;
	}

	PVSourceWorkspace* workspace = qobject_cast<PVSourceWorkspace*>(widget(index));
	assert(workspace);
	Inendi::PVRoot_sp root_sp = get_scene().get_parent<Inendi::PVRoot>().shared_from_this();
	PVHive::call<FUNC(Inendi::PVRoot::select_source)>(root_sp, *workspace->get_source());
}
