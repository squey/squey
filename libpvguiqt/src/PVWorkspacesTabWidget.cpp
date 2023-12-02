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
#include <pvguiqt/PVImportWorkflowTabBar.h>
#include <pvguiqt/PVErrorsAndWarnings.h>
#include <PVFormatBuilderWidget.h>

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
#include <QStackedWidget>

#define TAB_OPENING_EFFECT_MSEC 200

/******************************************************************************
 *
 * PVGuiQt::PVSceneTabBar
 *
 *****************************************************************************/

PVGuiQt::PVSceneTabBar::PVSceneTabBar(QWidget* parent /* = nullptr */)
    : QTabBar(parent)
{
	setTabsClosable(true);

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

// void PVGuiQt::PVSceneTabBar::resizeEvent(QResizeEvent* event)
// {
// 	QString stylesheet = "";

// 	if (count() > 0) {
// 		int width = _tab_widget->size().width() / count();

// 		// FIXME(pbrunet) : setting min_width to MIN_WIDTH and max_width to max(width, MIN_WIDTH)
// 		// should do the same
// 		if (width > MIN_WIDTH) {
// 			QFontMetrics metrics = QFontMetrics(font());
// 			for (int i = 0; i < count(); i++) {
// 				if (metrics.horizontalAdvance(tabText(i)) > width) {
// 					stylesheet = QString("QTabBar::tab { max-width: %1px; } ").arg(width);
// 					break;
// 				}
// 			}
// 			stylesheet += QString("QTabBar::tab { min-width: %1px; } ").arg(MIN_WIDTH);
// 		} else {
// 			stylesheet = QString("QTabBar::tab { width: %1px; } ").arg(MIN_WIDTH);
// 		}
// 		update();
// 	}
// 	_tab_widget->setStyleSheet(stylesheet);

// 	QTabBar::resizeEvent(event);
// }

/******************************************************************************
 *
 * PVGuiQt::PVSceneWorkspacesTabWidget
 *
 *****************************************************************************/
PVGuiQt::PVSceneWorkspacesTabWidget::PVSceneWorkspacesTabWidget(Squey::PVScene& scene,
                                                                QWidget* parent /* = 0 */)
    : QWidget(parent), _scene(scene)
{
	setObjectName("PVWorkspacesTabWidget");

	_import_worflow_tab_bar = new PVGuiQt::PVImportWorkflowTabBar();
	//_import_worflow_tab_bar->addTab("Data");
	_import_worflow_tab_bar->addTab("Format");
	_import_worflow_tab_bar->addTab("Errors");
	_import_worflow_tab_bar->addTab("Visualization");

	_workspace_tab_bar = new PVGuiQt::PVSceneTabBar(this);

	QWidget* tabbars_widget = new QWidget();
	QHBoxLayout* tabbars_layout = new QHBoxLayout;
	tabbars_layout->setContentsMargins(0, 0, 0, 0);
	tabbars_layout->addWidget(_import_worflow_tab_bar);
	tabbars_layout->addWidget(_workspace_tab_bar);
	tabbars_layout->addStretch();
	tabbars_widget->setLayout(tabbars_layout);

	_stacked_widget = new QStackedWidget();

	//_stacked_widget_data = new QStackedWidget();
	_stacked_widget_format = new QStackedWidget();
	_stacked_widget_errors = new QStackedWidget();
	_stacked_widget_workspace = new QStackedWidget();

	//_stacked_widget->addWidget(_stacked_widget_data);
	_stacked_widget->addWidget(_stacked_widget_format);
	_stacked_widget->addWidget(_stacked_widget_errors);
	_stacked_widget->addWidget(_stacked_widget_workspace);

	set_current_workflow_tab((int) EImportWorkflowStage::WORKSPACE);

	QVBoxLayout* layout = new QVBoxLayout();
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(tabbars_widget);
	layout->addWidget(_stacked_widget);

	setLayout(layout);

	// To get notified of mouse events we must enable mouse tracking on the QTabBar
	_import_worflow_tab_bar->setMouseTracking(true);
	_workspace_tab_bar->setMouseTracking(true);

	connect(_workspace_tab_bar, &QTabBar::currentChanged, this, &PVSceneWorkspacesTabWidget::tab_changed);
	connect(_workspace_tab_bar, &QTabBar::tabCloseRequested, this, &PVSceneWorkspacesTabWidget::tab_close_requested);

	connect(_import_worflow_tab_bar, &QTabBar::currentChanged, [this]() {
		_stacked_widget->setCurrentIndex(_import_worflow_tab_bar->currentIndex());
	});

	scene._project_updated.connect(
	    sigc::mem_fun(*this, &PVGuiQt::PVSceneWorkspacesTabWidget::set_project_modified));
}

static size_t invalid_columns_count(const Squey::PVSource* src)
{
	const PVRush::PVNraw& nraw = src->get_rushnraw();

	size_t invalid_columns_count = 0;
	for (PVCol col(0); col < nraw.column_count(); col++) {
		invalid_columns_count +=
		    bool(nraw.column(col).has_invalid() & pvcop::db::INVALID_TYPE::INVALID);
	}

	return invalid_columns_count;
}

void PVGuiQt::PVSceneWorkspacesTabWidget::show_errors_and_warnings()
{
	auto& cur_src = _scene.get_parent<Squey::PVRoot>().current_view()->get_parent<Squey::PVSource>();
	if (invalid_columns_count(&cur_src)) {
		_import_worflow_tab_bar->setTabEnabled((int) EImportWorkflowStage::ERRORS, true);
	}
}

void PVGuiQt::PVSceneWorkspacesTabWidget::set_current_tab(int index)
{
	_workspace_tab_bar->setCurrentIndex(index);
	_stacked_widget_workspace->setCurrentIndex(index);
}

void PVGuiQt::PVSceneWorkspacesTabWidget::set_current_workflow_tab(int index)
{
	_stacked_widget->setCurrentIndex(index);
	_import_worflow_tab_bar->setCurrentIndex(index);
}

QWidget* PVGuiQt::PVSceneWorkspacesTabWidget::current_widget()
{
	return _stacked_widget_workspace->currentWidget();
}

int PVGuiQt::PVSceneWorkspacesTabWidget::index_of(QWidget* workspace)
{
	return _stacked_widget_workspace->indexOf(workspace);
}



void PVGuiQt::PVSceneWorkspacesTabWidget::add_workspace(PVWorkspaceBase* workspace,
                                                        const QString& label)
{
	// Add the new workspace and select it
	int index = _workspace_tab_bar->addTab(label);
	_stacked_widget_workspace->addWidget(workspace);
	set_current_tab(index);


	// Add the format builder widget to its stacked widget // FIXME: what if more sources uses the same format ?
	auto& cur_src = _scene.get_parent<Squey::PVRoot>().current_view()->get_parent<Squey::PVSource>();
	PVRush::PVFormat const& format = cur_src.get_original_format();
	auto* editorWidget = new App::PVFormatBuilderWidget(this);
	if (not format.get_full_path().isEmpty()) {
		editorWidget->openFormat(format.get_full_path());
	}
	_stacked_widget_format->addWidget(editorWidget);
	_stacked_widget_format->setCurrentWidget(editorWidget);

	auto* source_workspace = dynamic_cast<PVGuiQt::PVSourceWorkspace*>(workspace);
	auto* d = source_workspace->get_source_invalid_evts_dlg();
	if (source_workspace->has_errors_or_warnings()) {
		PVErrorsAndWarnings* errors = new PVErrorsAndWarnings(source_workspace->get_source(), d);
		_stacked_widget_errors->addWidget(errors);
		_stacked_widget_errors->setCurrentWidget(errors);
	}
	else {
		QWidget* w = new QWidget;
		_stacked_widget_errors->addWidget(w);
		_stacked_widget_errors->setCurrentWidget(w);
	}


	// Add an animation on the tabBar.
	auto* animation = new QPropertyAnimation(this, "tab_width");
	animation->setDuration(TAB_OPENING_EFFECT_MSEC);
	animation->setStartValue(25);
	animation->setEndValue(_workspace_tab_bar->tabRect(index).width());
	animation->start(QAbstractAnimation::DeleteWhenStopped);

	connect(animation, &QPropertyAnimation::finished, this,
	        &PVSceneWorkspacesTabWidget::animation_finished);

	set_worflow_tab_status(index);
}

void PVGuiQt::PVSceneWorkspacesTabWidget::remove_workspace(int index)
{
	auto* animation = new QPropertyAnimation(this, "tab_width");
	blockSignals(true);
	set_current_tab(index); // Force current index in order to get the animation
	// on the selected tab!
	blockSignals(false);
	animation->setDuration(TAB_OPENING_EFFECT_MSEC);
	animation->setStartValue(_workspace_tab_bar->tabRect(index).width());
	animation->setEndValue(25);
	animation->start(QAbstractAnimation::DeleteWhenStopped);

	connect(animation, &QPropertyAnimation::finished, [this, index]() {
		_workspace_tab_bar->setStyleSheet("");
		QWidget* w = _stacked_widget_workspace->widget(index);
		_workspace_tab_bar->removeTab(index);
		_stacked_widget_workspace->removeWidget(w);
		_stacked_widget_format->removeWidget(_stacked_widget_format->widget(index));
		get_scene().remove_child(*qobject_cast<PVGuiQt::PVSourceWorkspace*>(w)->get_source());
		delete w;
		if (_workspace_tab_bar->count() == 0) {
			Q_EMIT is_empty();
			hide();
		} else {
			set_current_tab(std::min(index, _workspace_tab_bar->count() - 1));
		}
	});
}

void PVGuiQt::PVSceneWorkspacesTabWidget::set_tab_width(int tab_width)
{
	_workspace_tab_bar->setStyleSheet(QString("QTabBar::tab:selected { width: %1px;}").arg(tab_width));
}

void PVGuiQt::PVSceneWorkspacesTabWidget::animation_finished()
{
	_workspace_tab_bar->setStyleSheet("");
}

void PVGuiQt::PVSceneWorkspacesTabWidget::tab_close_requested(int index)
{
	remove_workspace(index);
}

// void PVGuiQt::PVSceneWorkspacesTabWidget::resizeEvent(QResizeEvent* event)
// {
// 	// FIXME(pbrunet) : Check if it is not done by default.
// 	dynamic_cast<PVSceneTabBar*>(_workspace_tab_bar)->resizeEvent(event);
// 	QTabWidget::resizeEvent(event);
// }

void PVGuiQt::PVSceneWorkspacesTabWidget::set_project_modified()
{
	if (!_project_modified) {
		Q_EMIT project_modified();
	}
	_project_modified = true;
}

void PVGuiQt::PVSceneWorkspacesTabWidget::tab_changed(int index)
{
	// TODO : should update current sub stacked widget index 
	if (index == -1 or _stacked_widget_workspace->count() == 0) {
		return;
	}

	for (QStackedWidget* stacked_widget : { /*_stacked_widget_data,*/ _stacked_widget_format, _stacked_widget_errors, _stacked_widget_workspace }) {
		stacked_widget->setCurrentIndex(index);
	}

	set_worflow_tab_status(index);

	auto* workspace = qobject_cast<PVSourceWorkspace*>(_stacked_widget_workspace->widget(index));
	assert(workspace);
	get_scene().get_parent<Squey::PVRoot>().select_source(*workspace->get_source());
}

void PVGuiQt::PVSceneWorkspacesTabWidget::set_worflow_tab_status(int index)
{
	auto* workspace = qobject_cast<PVSourceWorkspace*>(_stacked_widget_workspace->widget(index));
	_import_worflow_tab_bar->setTabEnabled((int) EImportWorkflowStage::ERRORS, dynamic_cast<PVGuiQt::PVSourceWorkspace*>(workspace)->has_errors_or_warnings());
	_import_worflow_tab_bar->setTabEnabled((int) EImportWorkflowStage::FORMAT, dynamic_cast<PVGuiQt::PVSourceWorkspace*>(workspace)->source_type() == "text");
}