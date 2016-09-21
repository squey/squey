/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVGUIQT_PVPROJECTSTABWIDGET_H__
#define __PVGUIQT_PVPROJECTSTABWIDGET_H__

#include <sigc++/sigc++.h>

#include <cassert>

#include <inendi/PVScene.h>
#include <inendi/PVRoot.h>

#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVWorkspace.h>

#include <QWidget>
#include <QObject>
#include <QTabWidget>
#include <QMouseEvent>
#include <QSize>
#include <QStackedWidget>
#include <QSplitterHandle>
#include <QSplitter>
#include <QTabBar>

namespace Inendi
{
class PVRoot;
}

namespace PVGuiQt
{

class PVStartScreenWidget;
class PVSceneWorkspacesTabWidget;

namespace __impl
{

class PVTabBar : public QTabBar
{
	Q_OBJECT

  public:
	explicit PVTabBar(Inendi::PVRoot& root, QWidget* parent = 0) : QTabBar(parent), _root(root) {}

  protected:
	void mouseDoubleClickEvent(QMouseEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;

  private Q_SLOTS:
	void rename_tab();
	void rename_tab(int index);

  private:
	Inendi::PVRoot& _root;
};

class PVTabWidget : public QTabWidget
{
  public:
	explicit PVTabWidget(Inendi::PVRoot& root, QWidget* parent = 0) : QTabWidget(parent)
	{
		setTabBar(new PVTabBar(root, this));
	}

  public:
	QTabBar* tabBar() const { return QTabWidget::tabBar(); }
};

class PVSplitterHandle : public QSplitterHandle
{
  public:
	explicit PVSplitterHandle(Qt::Orientation orientation, QSplitter* parent = 0)
	    : QSplitterHandle(orientation, parent)
	{
	}
	void set_max_size(int max_size) { _max_size = max_size; }
	int get_max_size() const { return _max_size; }

  protected:
	void mouseMoveEvent(QMouseEvent* event) override
	{
		assert(_max_size > 0); // set splitter handle max size!
		QList<int> sizes = splitter()->sizes();
		assert(sizes.size() > 0);
		if ((sizes[0] == 0 && event->pos().x() < _max_size) ||
		    (sizes[0] != 0 && event->pos().x() < 0)) {
			QSplitterHandle::mouseMoveEvent(event);
		}
	}

  private:
	int _max_size = 0;
};

class PVSplitter : public QSplitter
{
  public:
	explicit PVSplitter(Qt::Orientation orientation, QWidget* parent = 0)
	    : QSplitter(orientation, parent)
	{
	}

  protected:
	QSplitterHandle* createHandle() { return new PVSplitterHandle(orientation(), this); }
};
}

/**
 * \class PVProjectsTabWidget
 *
 * \note This class is representing a project tab widget.
 */
class PVProjectsTabWidget : public QWidget, public sigc::trackable
{
	Q_OBJECT

  public:
	static constexpr int FIRST_PROJECT_INDEX = 1;

  public:
	explicit PVProjectsTabWidget(Inendi::PVRoot* root, QWidget* parent = 0);

	PVSceneWorkspacesTabWidget* add_project(Inendi::PVScene& scene_p);
	void remove_project(PVSceneWorkspacesTabWidget* workspace_tab_widget);

	PVSourceWorkspace* add_source(Inendi::PVSource* source);

	void add_workspace(PVSourceWorkspace* workspace);
	void remove_workspace(PVSourceWorkspace* workspace);

	bool save_modified_projects();
	bool is_current_project_untitled() { return current_project() != nullptr; }
	void collapse_tabs(bool collapse = true);

	inline Inendi::PVScene* current_scene() const { return _root->current_scene(); }
	PVSceneWorkspacesTabWidget* current_workspace_tab_widget() const;
	inline PVSceneWorkspacesTabWidget* current_project() const
	{
		return (_current_workspace_tab_widget_index >= FIRST_PROJECT_INDEX)
		           ? (PVSceneWorkspacesTabWidget*)_stacked_widget->widget(
		                 _current_workspace_tab_widget_index)
		           : nullptr;
	}

	inline void select_tab_from_scene(Inendi::PVScene* scene);
	inline PVWorkspaceBase* current_workspace() const
	{
		return current_project() ? (PVWorkspaceBase*)current_project()->currentWidget() : nullptr;
	}
	inline Inendi::PVView* current_view() const { return _root->current_view(); }
	inline int projects_count() { return _tab_widget->count() - FIRST_PROJECT_INDEX; }
	inline const QStringList get_projects_list()
	{
		QStringList projects_list;
		for (int i = FIRST_PROJECT_INDEX; i < _tab_widget->count(); i++) {
			projects_list << _tab_widget->tabText(i);
		}
		return projects_list;
	}
	inline int get_current_project_index()
	{
		return _current_workspace_tab_widget_index - FIRST_PROJECT_INDEX;
	}
	Inendi::PVScene* get_scene_from_path(const QString& path);
	PVSceneWorkspacesTabWidget* get_workspace_tab_widget_from_scene(const Inendi::PVScene* scene);

  private Q_SLOTS:
	void current_tab_changed(int index);
	void emit_workspace_dragged_outside(QWidget* workspace)
	{
		Q_EMIT workspace_dragged_outside(workspace);
	}
	bool tab_close_requested(int index);
	void close_project();
	void project_modified();
	void select_tab_from_current_scene();

  Q_SIGNALS:
	void is_empty();
	void workspace_dragged_outside(QWidget* workspace);
	void new_project();
	void load_source_from_description(PVRush::PVSourceDescription);
	void load_project();
	void load_project_from_path(const QString& project);
	void load_source();
	void import_type(const QString&);
	void new_format();
	void load_format();
	void edit_format(const QString& format);
	void save_project();
	void active_project(bool active);

  private:
	bool maybe_save_project(int index);
	void create_unclosable_tabs();
	void remove_project(int index);

  private:
	__impl::PVSplitter* _splitter = nullptr;
	__impl::PVTabWidget* _tab_widget = nullptr; // QTabWidget has a problem with CSS and
	                                            // background-color, that's why this class isn't
	                                            // inheriting from QTabWidget...
	QStackedWidget* _stacked_widget = nullptr;
	PVStartScreenWidget* _start_screen_widget;
	int _current_workspace_tab_widget_index;
	Inendi::PVRoot* _root;
};
}

#endif /* __PVGUIQT_PVPROJECTSTABWIDGET_H__ */
