//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVWorkspace.h>

#include <squey/PVRoot.h>

#include <pvkernel/core/PVProgressBox.h>

#include <sigc++/sigc++.h>

#include <QDrag>
#include <QImage>
#include <QMimeData>
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
	connect(this, &QTabBar::currentChanged, _tab_widget, &PVSceneWorkspacesTabWidget::tab_changed);

	setMovable(true);
	setElideMode(Qt::ElideRight);
}

void PVGuiQt::PVSceneTabBar::mouseReleaseEvent(QMouseEvent* event)
{
	_drag_ongoing = false;

	// Tabs are closed on middle button click
	if (event->button() == Qt::MiddleButton) {
		int tab_index = tabAt(event->pos());
		if (tab_index < count()) {
			Q_EMIT tabCloseRequested(tab_index);
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
	auto* drag = new QDrag(this);

	auto* mimeData = new QMimeData;

	QByteArray byte_array;
	byte_array.reserve(sizeof(void*));
	byte_array.append((const char*)&workspace, sizeof(void*));

	mimeData->setData("application/x-squey_workspace", byte_array);

	drag->setMimeData(mimeData);

	// Set semi transparent thumbnail
	QImage opaque = workspace->grab(workspace->rect()).scaledToWidth(200).toImage();
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
		Q_EMIT _tab_widget->workspace_dragged_outside(workspace);
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
				if (metrics.horizontalAdvance(tabText(i)) > width) {
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
PVGuiQt::PVSceneWorkspacesTabWidget::PVSceneWorkspacesTabWidget(Squey::PVScene& scene,
                                                                QWidget* parent /* = 0 */)
    : QTabWidget(parent), _scene(scene)
{
	setObjectName("PVWorkspacesTabWidget");

	// To get notified of mouse events we must enable mouse tracking on *both*
	// QTabWidget and its underlying QTabBar
	setMouseTracking(true);
	tabBar()->setMouseTracking(true);

	scene._project_updated.connect(
	    sigc::mem_fun(*this, &PVGuiQt::PVSceneWorkspacesTabWidget::set_project_modified));

	setTabBar(new PVSceneTabBar(this));
}

void PVGuiQt::PVSceneWorkspacesTabWidget::add_workspace(PVWorkspaceBase* workspace,
                                                        const QString& label)
{
	// Add the new workspace and select it
	int index = addTab(workspace, label);
	setCurrentIndex(index);

	// Add an animation on the tabBar.
	auto* animation = new QPropertyAnimation(this, "tab_width");
	animation->setDuration(TAB_OPENING_EFFECT_MSEC);
	animation->setStartValue(25);
	animation->setEndValue(tabBar()->tabRect(index).width());
	animation->start(QAbstractAnimation::DeleteWhenStopped);

	connect(animation, &QPropertyAnimation::finished, this,
	        &PVSceneWorkspacesTabWidget::animation_finished);
}

void PVGuiQt::PVSceneWorkspacesTabWidget::remove_workspace(int index)
{
	auto* animation = new QPropertyAnimation(this, "tab_width");
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
		get_scene().remove_child(*qobject_cast<PVGuiQt::PVSourceWorkspace*>(w)->get_source());
		delete w;
		if (count() == 0) {
			Q_EMIT is_empty();
			hide();
		} else {
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
	remove_workspace(index);
}

void PVGuiQt::PVSceneWorkspacesTabWidget::resizeEvent(QResizeEvent* event)
{
	// FIXME(pbrunet) : Check if it is not done by default.
	dynamic_cast<PVSceneTabBar*>(tabBar())->resizeEvent(event);
	QTabWidget::resizeEvent(event);
}

void PVGuiQt::PVSceneWorkspacesTabWidget::set_project_modified()
{
	if (!_project_modified) {
		Q_EMIT project_modified();
	}
	_project_modified = true;
}

void PVGuiQt::PVSceneWorkspacesTabWidget::tab_changed(int index)
{
	if (index == -1) {
		return;
	}

	auto* workspace = qobject_cast<PVSourceWorkspace*>(widget(index));
	assert(workspace);
	get_scene().get_parent<Squey::PVRoot>().select_source(*workspace->get_source());
}
