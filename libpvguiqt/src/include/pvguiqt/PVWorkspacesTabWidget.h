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

class QComboBox;
class QLineEdit;
class QMouseEvent;
class QWidget;
#include <QAbstractAnimation>
#include <QObject>
#include <QPoint>
#include <QTabBar>
#include <QTabWidget>
#include <QTimer>

namespace Picviz
{
class PVSource;
class PVScene;
#ifdef ENABLE_CORRELATION
class PVAD2GView;
#endif
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
	/**
	 * \class PVSaveSceneToFileFuncObserver
	 *
	 * \note This class is handling "*" for modified scene workspaces.
	 */
	class PVSaveSceneToFileFuncObserver: public PVHive::PVFuncObserverSignal<Picviz::PVScene, FUNC(Picviz::PVScene::save_to_file)>
	{
	public:
		PVSaveSceneToFileFuncObserver(PVSceneWorkspacesTabWidget* parent) : _parent(parent) {}
	public:
		void update(const arguments_deep_copy_type& args) const;
	private:
		PVSceneWorkspacesTabWidget* _parent;
	};

	/**
	 * \class TabRenamerEventFilter
	 *
	 * \note This class is handling in-place tab name editing for open workspaces.
	 */
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
}

/**
 * \class PVSceneTabBar
 *
 * \note This class is handling tab bar event for PVSceneWorkspacesTabWidget.
 */
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

/**
 * \class PVOpenWorkspaceTabBar
 *
 * \note This class is a PVSceneTabBar derivation handling tab bar event for PVOpenWorkspacesTabWidget.
 */
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

/**
 * \class PVWorkspacesTabWidgetBase
 *
 * \note This class is the base class for workspaces tab widgets.
 */
class PVWorkspacesTabWidgetBase : public QTabWidget
{
	Q_OBJECT
	Q_PROPERTY(int tab_width READ get_tab_width WRITE set_tab_width);

	friend class PVSceneTabBar;
	friend class PVOpenWorkspaceTabBar;

public:
	PVWorkspacesTabWidgetBase(Picviz::PVRoot& root, QWidget* parent = 0);

public:
	/*! \brief Add a workspace with or without animation.
	 */
	virtual int add_workspace(PVWorkspaceBase* page, const QString & label, bool animation = true);

	/*! \brief Remove a workspace with or without animation.
	 */
	virtual void remove_workspace(int index, bool animation = true);

#ifdef ENABLE_CORRELATION
	/*! \brief Return the current correlation of the workspace.
	 */
	virtual Picviz::PVAD2GView* get_correlation() = 0;

	/*! \brief Return a list of available correlations for the workspace.
	 */
	virtual Picviz::PVRoot::correlations_t get_correlations() = 0;
#endif

	/*! \brief Returns the number of affective tabs in the widget (ie: special tab "+" button is not taken into account).
	 */
	int count() const { return _tab_bar->count(); }

	QList<PVWorkspaceBase*> list_workspaces() const;

protected:
#ifdef ENABLE_CORRELATION
	/*! \brief Returns the index of a given correlation in the correlations combo box.
	 */
	int get_index_from_correlation(void* correlation);
#endif

	inline Picviz::PVRoot const& get_root() const { return _root; }
	inline Picviz::PVRoot& get_root() { return _root; }

signals:
	/*! \brief Signal emitted when a workspace is dragged outside of a PVMainWindow.
	 */
	void workspace_dragged_outside(QWidget*);

	/*! \brief Signal emitted when the tab animation is finished.
	 */
	void animation_finished();

protected slots:
	/*! \brief Slot called when the user closes a workspace.
	 */
	void tab_close_requested(int index);

#ifdef ENABLE_CORRELATION
	/*! \brief Slot called when the user changes the current correlation from the combo box.
	 */
	virtual void correlation_changed(int index) = 0;

	/*! \brief Keep the sync with available correlations.
	 *  \note For source workspaces, a correlation is available if one of its views belong to the source.
	 */
	virtual void update_correlations_list();
#endif

private slots:
	/*! \brief Change the CSS property width of the selected tab (used by the animation).
	 */
	void set_tab_width(int tab_width);

	/*! \brief Dummy slot to avoid warning with the tab_width Q_PROPERTY.
	 */
	int get_tab_width() const { return 0; }

	/*! \brief Emit "animation_finished" signal when the animation finished.
	 */
	void animation_state_changed(QAbstractAnimation::State new_state, QAbstractAnimation::State old_state);

protected:
	QComboBox* _combo_box;
	PVSceneTabBar* _tab_bar;

private:
	int _tab_animated_width;
	bool _tab_animation_ongoing = false;
	int _tab_animation_index;
	
	PVHive::PVObserverSignal<Picviz::PVRoot>* _obs;
	Picviz::PVRoot& _root;
};

/**
 * \class PVSceneWorkspacesTabWidget
 *
 * \note This class is a PVWorkspacesTabWidgetBase derivation representing a scene tab widget.
 */
class PVSceneWorkspacesTabWidget : public PVWorkspacesTabWidgetBase
{
	Q_OBJECT
	friend class __impl::PVSaveSceneToFileFuncObserver;
	friend class PVSceneTabBar;

public:
	PVSceneWorkspacesTabWidget(Picviz::PVScene& scene, QWidget* parent = 0);

public:

	/*! \brief Remove the workspace and close its associated source if needed.
     */
	void remove_workspace(int index, bool close_source = true) override;

#ifdef ENABLE_CORRELATION
	/*! \brief Return the current correlation of the workspace.
	 */
	Picviz::PVAD2GView* get_correlation() override { return _correlation; }

	/*! \brief Return a list of available correlations for the workspace (using PVAD2GView::get_used_views(PVScene*)).
	 */
	Picviz::PVRoot::correlations_t get_correlations() override { return get_root().get_correlations_for_scene(*get_scene()); }
#endif

	bool is_project_modified() { return _project_modified; }
	bool is_project_untitled() { return _project_untitled; }

	QList<Picviz::PVSource*> list_sources() const;
	Picviz::PVScene* get_scene() { return _obs_scene.get_object(); }

protected:
	/*! \brief Special behavior on workspace removal (emit "is_empty" when all sources are closed).
	 */
	void tabRemoved(int index) override;

signals:
	void project_modified(bool, QString = QString());

	/*! \brief Signal emitted when the last source is closed.
	 */
	void is_empty();

public slots:
    /*! \brief Call Picviz::PVRoot::select_source throught the Hive to keep track of current source.
	 */
	void tab_changed(int index);

protected slots:
#ifdef ENABLE_CORRELATION
	/*! \brief Slot called when the user changes the current correlation from the combo box.
	 */
	void correlation_changed(int index);
#endif

	void check_new_sources();

private slots:
	void set_project_modified(bool modified = true, QString path = QString());

private:
	bool _project_modified = false;
	bool _project_untitled = true;

#ifdef ENABLE_CORRELATION
	Picviz::PVAD2GView* _correlation = nullptr;
#endif

	PVHive::PVObserverSignal<Picviz::PVScene> _obs_scene;
	__impl::PVSaveSceneToFileFuncObserver _save_scene_func_observer;
};

/**
 * \class PVOpenWorkspacesTabWidget
 *
 * \note This class is a PVWorkspacesTabWidgetBase derivation representing an open workspace tab widget.
 */
class PVOpenWorkspacesTabWidget : public PVWorkspacesTabWidgetBase
{
	Q_OBJECT
	friend class PVOpenWorkspaceTabBar;

public:
	PVOpenWorkspacesTabWidget(Picviz::PVRoot& root, QWidget* parent = 0);

public:
#ifdef ENABLE_CORRELATION
	Picviz::PVAD2GView* get_correlation() override;
	Picviz::PVRoot::correlations_t get_correlations() override { return get_root().get_correlations(); }
#endif

	PVOpenWorkspace* current_workspace() const;
	PVOpenWorkspace* current_workspace_or_create();

protected:
	/*! \brief Special behavior on workspace creation (handle drag&drop tab switch).
	 */
	void tabInserted(int index) override;

	/*! \brief Special behavior on workspace removal (emit "is_empty" signal when the last source is closed).
	 */
	void tabRemoved(int index) override;

public slots:
    /*! \brief Special behavior on tab change: select current correlation.
	 */
	void tab_changed(int index);

#ifdef ENABLE_CORRELATION
protected slots:
	/*! \brief Slot called when the user changes the current correlation from the combo box.
	 */
	void correlation_changed(int index);
#endif

public slots:
	/*! \brief Slot called to check if display drag&drop needs to switch tab.
	 */
	void start_checking_for_automatic_tab_switch();

	/*! \brief Slot called to switch tab based on the cursor position.
	 */
	void switch_tab();

private:
	QTimer _automatic_tab_switch_timer;
	int _tab_switch_index;
};

}

#endif // __PVGUIQT_PVWORKSPACESTABWIDGET_H__
