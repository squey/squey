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

#include <pvguiqt/PVProjectsTabWidget.h>
#include <pvguiqt/PVStartScreenWidget.h>

#include <QHBoxLayout>
#include <QMessageBox>
#include <QInputDialog>
#include <QMenu>
#include <QAction>

const QString star = "*";

/******************************************************************************
 *
 * PVGuiQt::__impl::PVTabBar
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVTabBar::mouseDoubleClickEvent(QMouseEvent* event)
{
	int index = tabAt(event->pos());
	if (index >= PVProjectsTabWidget::FIRST_PROJECT_INDEX) {
		rename_tab(index);
	}
	QTabBar::mouseDoubleClickEvent(event);
}

void PVGuiQt::__impl::PVTabBar::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::RightButton) {
		int index = tabAt(event->pos());
		if (index >= PVProjectsTabWidget::FIRST_PROJECT_INDEX) {
			auto* menu = new QMenu(this);
			QAction* rename_action = menu->addAction("&Rename...");
			rename_action->setData(QVariant::fromValue(index));
			connect(rename_action, SIGNAL(triggered(bool)), this, SLOT(rename_tab()));
			menu->popup(event->globalPosition().toPoint());
		}
	}
	QTabBar::mousePressEvent(event);
}

void PVGuiQt::__impl::PVTabBar::keyPressEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_F2) {
		int index = currentIndex();
		if (index >= PVProjectsTabWidget::FIRST_PROJECT_INDEX) {
			rename_tab(index);
		}
	}
	QTabBar::keyPressEvent(event);
}

void PVGuiQt::__impl::PVTabBar::rename_tab()
{
	auto* rename_action = (QAction*)sender();
	assert(rename_action);
	int index = rename_action->data().toInt();
	rename_tab(index);
}

void PVGuiQt::__impl::PVTabBar::rename_tab(int index)
{
	QString tab_name = tabText(index);
	bool add_star = false;
	if (tab_name.endsWith(star)) {
		add_star = true;
		tab_name = tab_name.left(tab_name.size() - 1);
	}
	QString name = QInputDialog::getText(this, "Rename data collection",
	                                     "New data collection name:", QLineEdit::Normal, tab_name);
	if (!name.isEmpty()) {
		setTabText(index, name + (add_star ? star : ""));
		Squey::PVScene* scene = _root.current_scene();
		assert(scene);
		scene->set_name(name.toStdString());
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVProjectsTabWidget
 *
 *****************************************************************************/
PVGuiQt::PVProjectsTabWidget::PVProjectsTabWidget(Squey::PVRoot* root, QWidget* parent /*= 0*/)
    : QWidget(parent), _root(root)
{
	assert(root);
	setObjectName("PVProjectsTabWidget");

	auto* main_layout = new QHBoxLayout();

	_tab_widget = new __impl::PVTabWidget(*root);
	_tab_widget->setTabsClosable(true);

	_stacked_widget = new QStackedWidget();

	create_unclosable_tabs();

	_splitter = new __impl::PVSplitter(Qt::Horizontal);
	_splitter->setChildrenCollapsible(true);
	_splitter->addWidget(_tab_widget);
	_splitter->addWidget(_stacked_widget);
	_splitter->setStretchFactor(0, 0);
	_splitter->setStretchFactor(1, 1);
	int tab_width = _tab_widget->tabBar()->tabRect(0).width();
	((__impl::PVSplitterHandle*)_splitter->handle(1))->set_max_size(tab_width);
	QList<int> sizes;
	sizes << 1 << 2;
	_splitter->setSizes(sizes);

	main_layout->addWidget(_splitter);

	setLayout(main_layout);

	connect(_tab_widget, &QTabWidget::currentChanged, this,
	        &PVProjectsTabWidget::current_tab_changed);
	connect(_tab_widget->tabBar(), &QTabBar::tabCloseRequested, this,
	        &PVProjectsTabWidget::tab_close_requested);

	root->_scene_updated.connect(
	    sigc::mem_fun(*this, &PVGuiQt::PVProjectsTabWidget::select_tab_from_current_scene));
}

void PVGuiQt::PVProjectsTabWidget::create_unclosable_tabs()
{
	// Start screen widget
	_start_screen_widget = new PVGuiQt::PVStartScreenWidget();
	_tab_widget->addTab(new QWidget(), "");
	_tab_widget->setTabPosition(QTabWidget::West);
	_tab_widget->tabBar()->tabButton(0, QTabBar::RightSide)->resize(0, 0);

	QPixmap pm(":/squey");
	QTransform trans;
	_tab_widget->setTabIcon(0, pm.transformed(trans.rotate(90)));

	_tab_widget->setTabToolTip(0, "Start screen");
	_stacked_widget->addWidget(_start_screen_widget);
	connect(_start_screen_widget, SIGNAL(load_source_from_description(PVRush::PVSourceDescription)),
	        this, SIGNAL(load_source_from_description(PVRush::PVSourceDescription)));
	connect(_start_screen_widget, &PVStartScreenWidget::new_project, this,
	        &PVProjectsTabWidget::new_project);
	connect(_start_screen_widget, &PVStartScreenWidget::load_project, this,
	        &PVProjectsTabWidget::load_project);
	connect(_start_screen_widget, &PVStartScreenWidget::load_project_from_path, this,
	        &PVProjectsTabWidget::load_project_from_path);
	connect(_start_screen_widget, SIGNAL(import_type(const QString&)), this,
	        SIGNAL(import_type(const QString&)));
	connect(_start_screen_widget, &PVStartScreenWidget::new_format, this,
	        &PVProjectsTabWidget::new_format);
	connect(_start_screen_widget, &PVStartScreenWidget::load_format, this,
	        &PVProjectsTabWidget::load_format);
	connect(_start_screen_widget, &PVStartScreenWidget::edit_format, this,
	        &PVProjectsTabWidget::edit_format);
}

void PVGuiQt::PVProjectsTabWidget::collapse_tabs(bool collapse /* = true */)
{
	int max_size = ((__impl::PVSplitterHandle*)_splitter->handle(1))->get_max_size();
	QList<int> sizes;
	sizes << (collapse ? 0 : max_size) << 1;
	_splitter->setSizes(sizes);
}

void PVGuiQt::PVProjectsTabWidget::show_errors_and_warnings()
{
	auto* workspace_tab_widget = qobject_cast<PVSceneWorkspacesTabWidget*>(_stacked_widget->widget(_tab_widget->count()-1));
	workspace_tab_widget->show_errors_and_warnings();
}

PVGuiQt::PVSceneWorkspacesTabWidget*
PVGuiQt::PVProjectsTabWidget::add_project(Squey::PVScene& scene_p)
{
	auto* workspace_tab_widget = new PVSceneWorkspacesTabWidget(scene_p);
	workspace_tab_widget->setObjectName("workspace_tab_widget");
	connect(workspace_tab_widget, &PVSceneWorkspacesTabWidget::workspace_dragged_outside, this,
	        &PVProjectsTabWidget::emit_workspace_dragged_outside);
	connect(workspace_tab_widget, &PVSceneWorkspacesTabWidget::is_empty, this,
	        &PVProjectsTabWidget::close_project);
	connect(workspace_tab_widget, &PVSceneWorkspacesTabWidget::project_modified, this,
	        &PVProjectsTabWidget::project_modified);

	int index = _tab_widget->count();
	_tab_widget->insertTab(index, new QWidget(), QString::fromStdString(scene_p.get_name()));

	_stacked_widget->insertWidget(index, workspace_tab_widget);


	_tab_widget->setTabToolTip(index, QString::fromStdString(scene_p.get_name()));
	_tab_widget->setCurrentIndex(index);

	return workspace_tab_widget;
}

void PVGuiQt::PVProjectsTabWidget::project_modified()
{
	auto* workspace_tab_widget = (PVSceneWorkspacesTabWidget*)sender();
	assert(workspace_tab_widget);
	int index = _stacked_widget->indexOf(workspace_tab_widget);
	QString text = _tab_widget->tabText(index);
	if (!text.endsWith(star)) {
		_tab_widget->setTabText(index, text + "*");
	}
}

bool PVGuiQt::PVProjectsTabWidget::save_modified_projects()
{
	for (int i = FIRST_PROJECT_INDEX; i < _tab_widget->count(); i++) {
		auto* tab_widget =
		    (PVSceneWorkspacesTabWidget*)_stacked_widget->widget(i);
		if (tab_widget->is_project_modified()) {
			if (!tab_close_requested(i)) {
				return false;
			}
		}
	}

	return true;
}

void PVGuiQt::PVProjectsTabWidget::close_project()
{
	auto* workspace_tab_widget = (PVSceneWorkspacesTabWidget*)sender();
	assert(workspace_tab_widget);
	int index = _stacked_widget->indexOf(workspace_tab_widget);
	remove_project(index);
}

bool PVGuiQt::PVProjectsTabWidget::tab_close_requested(int index)
{
	QMessageBox ask(QMessageBox::Question, tr("Close data collection?"),
	                tr("Are you sure you want to close \"%1\"").arg(_tab_widget->tabText(index)),
	                QMessageBox::Yes | QMessageBox::No);
	if (ask.exec() == QMessageBox::Yes) {
		remove_project(index);
		return false;
	}
	return true;
}

PVGuiQt::PVSourceWorkspace* PVGuiQt::PVProjectsTabWidget::add_source(Squey::PVSource* source)
{
	auto* workspace = new PVGuiQt::PVSourceWorkspace(source);

	add_workspace(workspace);

	return workspace;
}

void PVGuiQt::PVProjectsTabWidget::add_workspace(PVSourceWorkspace* workspace)
{
	auto& scene = workspace->get_source()->get_parent<Squey::PVScene>();
	PVSceneWorkspacesTabWidget* workspace_tab_widget = get_workspace_tab_widget_from_scene(&scene);

	if (!workspace_tab_widget) {
		workspace_tab_widget = add_project(scene);
	}

	static constexpr const size_t MAX_LEN = 30;

	const Squey::PVSource* src = workspace->get_source();
	std::string tab_name = src->get_name();
	if (tab_name.size() > MAX_LEN) {
		tab_name = tab_name.substr(0, MAX_LEN) + "...";
	}
	workspace_tab_widget->add_workspace(workspace, QString::fromStdString(tab_name));

	int index = workspace_tab_widget->index_of(workspace);
	//workspace_tab_widget->setTabToolTip(index, src->get_tooltip());
	workspace_tab_widget->set_current_tab(index);

	_tab_widget->setCurrentIndex(_stacked_widget->indexOf(workspace_tab_widget));
}

void PVGuiQt::PVProjectsTabWidget::remove_workspace(PVSourceWorkspace* workspace)
{
	auto& scene = workspace->get_source()->get_parent<Squey::PVScene>();
	PVSceneWorkspacesTabWidget* workspace_tab_widget = get_workspace_tab_widget_from_scene(&scene);
	workspace_tab_widget->remove_workspace(workspace_tab_widget->index_of(workspace));
}

void PVGuiQt::PVProjectsTabWidget::remove_project(PVSceneWorkspacesTabWidget* workspace_tab_widget)
{
	remove_project(_stacked_widget->indexOf(workspace_tab_widget));
}

void PVGuiQt::PVProjectsTabWidget::remove_project(int index)
{
	if (index != -1) {
		auto* tab_widget =
		    (PVSceneWorkspacesTabWidget*)_stacked_widget->widget(index);
		tab_widget->get_scene().remove_from_tree();
		_tab_widget->removeTab(index);
		_stacked_widget->removeWidget(tab_widget);

		tab_widget->deleteLater();

		if (_tab_widget->count() == FIRST_PROJECT_INDEX) {
			_tab_widget->setCurrentIndex(0);
			Q_EMIT is_empty();
		}
	}
}

void PVGuiQt::PVProjectsTabWidget::current_tab_changed(int index)
{
	_stacked_widget->setCurrentIndex(
	    index); // Map QTabBar signal to QStackedWidget to keep the sync
	_current_workspace_tab_widget_index = index;

	// to report if the active tab is for a project or not (start page or empty
	// worspaces page)
	QWidget* active_widget = _stacked_widget->currentWidget();
	if ((active_widget == _start_screen_widget) ||
	    (_stacked_widget->count() <= FIRST_PROJECT_INDEX)) {
		Q_EMIT active_project(false);
	} else {
		const auto& scenes_list = _root->get_children<Squey::PVScene>();
		auto it = scenes_list.begin();
		std::advance(it, index - 1);
		_root->select_scene(**it);
		Q_EMIT active_project(true);
	}

	if (index == 0) {
		return;
	}
}

PVGuiQt::PVSceneWorkspacesTabWidget*
PVGuiQt::PVProjectsTabWidget::current_workspace_tab_widget() const
{
	if (_current_workspace_tab_widget_index < 0) {
		return nullptr;
	}

	QWidget* w = _stacked_widget->widget(_current_workspace_tab_widget_index);

	return qobject_cast<PVSceneWorkspacesTabWidget*>(w);
}

PVGuiQt::PVSceneWorkspacesTabWidget*
PVGuiQt::PVProjectsTabWidget::get_workspace_tab_widget_from_scene(const Squey::PVScene* scene)
{
	for (int i = FIRST_PROJECT_INDEX; i < _stacked_widget->count(); i++) {
		auto* workspace_tab_widget =
		    (PVSceneWorkspacesTabWidget*)_stacked_widget->widget(i);
		if (&workspace_tab_widget->get_scene() == scene) {
			return workspace_tab_widget;
		}
	}
	return nullptr;
}

void PVGuiQt::PVProjectsTabWidget::select_tab_from_scene(Squey::PVScene* scene)
{
	_tab_widget->setCurrentIndex(_tab_widget->indexOf(get_workspace_tab_widget_from_scene(scene)));
}

void PVGuiQt::PVProjectsTabWidget::select_tab_from_current_scene()
{
	Squey::PVScene* cur_scene = _root->current_scene();
	if (&cur_scene->get_parent<Squey::PVRoot>() != _root) {
		return;
	}

	select_tab_from_scene(cur_scene);
}
