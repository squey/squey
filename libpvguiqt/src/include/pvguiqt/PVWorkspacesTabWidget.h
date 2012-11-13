/**
 * \file PVWorkspacesTabWidget.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVWORKSPACESTABWIDGET_H__
#define __PVGUIQT_PVWORKSPACESTABWIDGET_H__

#include <picviz/PVScene.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVObserverSignal.h>
#include <pvhive/PVFuncObserver.h>

#include <pvkernel/core/lambda_connect.h>

#include <QApplication>
#include <QTabWidget>
#include <QTimer>
#include <QTabBar>
#include <QWidget>
#include <QMouseEvent>
#include <QPoint>
#include <QObject>
#include <QDrag>
#include <QLineEdit>
#include <QPropertyAnimation>
#include <QComboBox>

#include <iostream>

namespace Picviz
{
class PVSource;
class PVScene;
class PVAD2GView;
}

namespace PVGuiQt
{

class PVOpenWorkspace;
class PVWorkspaceBase;
class PVWorkspacesTabWidgetBase;
class PVSceneWorkspacesTabWidget;
class PVOpenWorkspaceTabBar;
class PVOpenWorkspacesTabWidget;
class PVSceneTabBar;

namespace __impl
{
	class PVSaveSceneToFileFuncObserver: public PVHive::PVFuncObserverSignal<Picviz::PVScene, FUNC(Picviz::PVScene::save_to_file)>
	{
	public:
		PVSaveSceneToFileFuncObserver(PVSceneWorkspacesTabWidget* parent) : _parent(parent) {}
	public:
		void update(const arguments_deep_copy_type& args) const;
	private:
		PVSceneWorkspacesTabWidget* _parent;
	};
}

class TabRenamerEventFilter : public QObject
{
public:
	TabRenamerEventFilter(PVGuiQt::PVOpenWorkspaceTabBar* tab_bar, int index, QLineEdit* line_edit) : _tab_bar(tab_bar), _index(index), _line_edit(line_edit) {}

	bool eventFilter(QObject* watched, QEvent* event);
private:
	PVGuiQt::PVOpenWorkspaceTabBar* _tab_bar;
	int _index;
	QLineEdit* _line_edit;
};

class PVSceneTabBar : public QTabBar
{
	Q_OBJECT

public:
	PVSceneTabBar(PVWorkspacesTabWidgetBase* tab_widget);
	QSize tabSizeHint(int index) const;
	virtual int count() const;

public:
	virtual PVOpenWorkspace* create_new_workspace() { return nullptr; }

protected:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void leaveEvent(QEvent* even) override;

	void start_drag(QWidget* workspace);

protected:
	PVWorkspacesTabWidgetBase* _tab_widget;
	QPoint _drag_start_position;
	bool _drag_ongoing = false;
};

class PVOpenWorkspaceTabBar : public PVSceneTabBar
{
	Q_OBJECT

public:
	PVOpenWorkspaceTabBar(PVOpenWorkspacesTabWidget* tab_widget);
	int count() const;
	PVOpenWorkspace* create_new_workspace() override;

protected:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent* event) override;
	void wheelEvent(QWheelEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;

private:
	int _workspace_id = 0;
};

/******************************************************************************
 *
 * PVGuiQt::PVWorkspacesTabWidgetBase
 *
 *****************************************************************************/
class PVWorkspacesTabWidgetBase : public QTabWidget
{
	Q_OBJECT
	Q_PROPERTY(int tab_width READ get_tab_width WRITE set_tab_width);
	friend class PVSceneTabBar;
	friend class PVOpenWorkspaceTabBar;

public:
	PVWorkspacesTabWidgetBase(Picviz::PVRoot& root, QWidget* parent = 0);

public:
	virtual Picviz::PVAD2GView* get_correlation() = 0;
	virtual Picviz::PVRoot::correlations_t get_correlations() = 0;
	virtual void remove_workspace(int index, bool close_animation = true);
	int addTab(PVWorkspaceBase* page, const QString & label);
	int count() const { return _tab_bar->count(); }

	QList<PVWorkspaceBase*> list_workspaces() const;

protected:
	int get_index_from_correlation(void* correlation);

signals:
	void workspace_dragged_outside(QWidget*);
	void workspace_closed(Picviz::PVSource* source);
	void animation_finished();

protected slots:
	virtual void correlation_changed(int index) = 0;
	void tabCloseRequested_Slot(int index);
	virtual void update_correlations_list();

private slots:
	int get_tab_width() const { return 0; }
	void set_tab_width(int tab_width);
	void animation_state_changed(QAbstractAnimation::State new_state, QAbstractAnimation::State old_state);

private:
	void emit_workspace_dragged_outside(QWidget* workspace) { emit workspace_dragged_outside(workspace); }

protected:
	QComboBox* _combo_box;
	PVSceneTabBar* _tab_bar;

	inline Picviz::PVRoot const& get_root() const { return _root; }
	inline Picviz::PVRoot& get_root() { return _root; }

private:
	int _tab_animated_width;
	bool _tab_animation_ongoing = false;
	int _tab_animation_index;
	
	PVHive::PVObserverSignal<Picviz::PVRoot>* _obs;
	Picviz::PVRoot& _root;
};

/******************************************************************************
 *
 * PVGuiQt::PVWorkspacesTabWidget
 *
 *****************************************************************************/
class PVSceneWorkspacesTabWidget : public PVWorkspacesTabWidgetBase
{
	Q_OBJECT
	friend class __impl::PVSaveSceneToFileFuncObserver;
	friend class PVSceneTabBar;

public:
	PVSceneWorkspacesTabWidget(Picviz::PVScene& scene, QWidget* parent = 0);

public:
	Picviz::PVScene* get_scene() { return _obs_scene.get_object(); }
	bool is_project_modified() { return _project_modified; }
	bool is_project_untitled() { return _project_untitled; }
	Picviz::PVAD2GView* get_correlation() override { return _correlation; }
	Picviz::PVRoot::correlations_t get_correlations() override { return get_root().get_correlations_for_scene(*get_scene()); }
	void remove_workspace(int index, bool close_animation = true) override;

	QList<Picviz::PVSource*> list_sources() const;
protected:
	void tabRemoved(int index) override;

signals:
	void project_modified(bool, QString = QString());
	void is_empty();

public slots:
	void tab_changed(int index);

protected slots:
	void correlation_changed(int index);
	void check_new_sources();

private slots:
	void set_project_modified(bool modified = true, QString path = QString());

private:
	bool _project_modified = false;
	bool _project_untitled = true;

	Picviz::PVAD2GView* _correlation = nullptr;

	PVHive::PVObserverSignal<Picviz::PVScene> _obs_scene;
	__impl::PVSaveSceneToFileFuncObserver _save_scene_func_observer;
};

/******************************************************************************
 *
 * PVGuiQt::PVOpenWorkspacesTabWidget
 *
 *****************************************************************************/
class PVOpenWorkspacesTabWidget : public PVWorkspacesTabWidgetBase
{
	Q_OBJECT
	friend class PVOpenWorkspaceTabBar;

public:
	PVOpenWorkspacesTabWidget(Picviz::PVRoot& root, QWidget* parent = 0);

public:
	Picviz::PVAD2GView* get_correlation() override;
	Picviz::PVRoot::correlations_t get_correlations() override { return get_root().get_correlations(); }
	PVOpenWorkspace* current_workspace() const;
	PVOpenWorkspace* current_workspace_or_create();

protected:
	void tabInserted(int index) override;
	void tabRemoved(int index) override;

public slots:
	void tab_changed(int index);

protected slots:
	void correlation_changed(int index);

public slots:
	void start_checking_for_automatic_tab_switch();
	void switch_tab();

private:
	QTimer _automatic_tab_switch_timer;
	int _tab_switch_index;
};

}

#endif // __PVGUIQT_PVWORKSPACESTABWIDGET_H__
