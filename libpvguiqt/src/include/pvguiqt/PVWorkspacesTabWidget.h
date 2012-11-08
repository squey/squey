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
}

namespace PVGuiQt
{

class PVOpenWorkspace;
class PVWorkspaceBase;
class PVWorkspacesTabWidgetBase;
class PVSceneWorkspacesTabWidget;
class PVOpenWorkspaceTabBar;
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
	virtual void create_new_workspace() {}

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
	PVOpenWorkspaceTabBar(PVWorkspacesTabWidgetBase* tab_widget) : PVSceneTabBar(tab_widget) {}
	int count() const;
	void create_new_workspace() override;

protected:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent* event) override;
	void wheelEvent(QWheelEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;
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
	PVWorkspacesTabWidgetBase(QWidget* parent = 0);

public:
	virtual int get_correlation_index() = 0;
	virtual void remove_workspace(int index, bool close_animation = true);
	int addTab(PVWorkspaceBase* page, const QString & label);
	int count() const { return _tab_bar->count(); }

protected:
	void tabInserted(int index) override;

signals:
	void workspace_dragged_outside(QWidget*);
	void workspace_closed(Picviz::PVSource* source);
	void animation_finished();

public slots:
	virtual void tab_changed(int index) = 0;

protected slots:
	virtual void correlation_changed(int index) = 0;
	void tabCloseRequested_Slot(int index);
	void update_correlations_list();

private slots:
	void start_checking_for_automatic_tab_switch();
	void switch_tab();
	int get_tab_width() const { return 0; }
	void set_tab_width(int tab_width);
	void animation_state_changed(QAbstractAnimation::State new_state, QAbstractAnimation::State old_state);

private:
	void emit_workspace_dragged_outside(QWidget* workspace) { emit workspace_dragged_outside(workspace); }

protected:
	QComboBox* _combo_box;
	PVSceneTabBar* _tab_bar;

private:
	QTimer _automatic_tab_switch_timer;
	int _tab_animated_width;
	bool _tab_animation_ongoing = false;
	int _tab_index;
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
	PVSceneWorkspacesTabWidget(Picviz::PVScene_p scene_p, QWidget* parent = 0);

public:
	Picviz::PVScene* get_scene() { return _scene_p.get(); }
	bool is_project_modified() { return _project_modified; }
	bool is_project_untitled() { return _project_untitled; }
	int get_correlation_index() override { return std::max(-1, _combo_box->findText(_correlation_name)-1); }
	void remove_workspace(int index, bool close_animation = true) override;

protected:
	void tabRemoved(int index) override;

signals:
	void project_modified(bool, QString = QString());
	void is_empty();

public slots:
	void tab_changed(int index);

protected slots:
	void correlation_changed(int index);

private slots:
	void set_project_modified(bool modified = true, QString path = QString());

private:
	Picviz::PVScene_p _scene_p;
	bool _project_modified = false;
	bool _project_untitled = true;

	QString _correlation_name;

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
	PVOpenWorkspacesTabWidget(QWidget* parent = 0);

public:
	int get_correlation_index() override;
	PVOpenWorkspace* current_workspace() const;

protected:
	void tabRemoved(int index) override;

public slots:
	void tab_changed(int index);

protected slots:
	void correlation_changed(int index);
};

}

#endif // __PVGUIQT_PVWORKSPACESTABWIDGET_H__
