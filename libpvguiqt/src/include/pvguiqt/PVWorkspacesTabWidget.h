/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVGUIQT_PVWORKSPACESTABWIDGET_H__
#define __PVGUIQT_PVWORKSPACESTABWIDGET_H__

#include <inendi/PVScene.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVObserverSignal.h>
#include <pvhive/PVFuncObserver.h>

#include <pvkernel/core/lambda_connect.h>

class QLineEdit;
class QMouseEvent;
class QWidget;
#include <QAbstractAnimation>
#include <QObject>
#include <QPoint>
#include <QTabBar>
#include <QTabWidget>
#include <QTimer>

namespace Inendi
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
class PVOpenWorkspacesTabWidget;
class PVSceneTabBar;

namespace __impl
{

/**
 * \class TabRenamerEventFilter
 *
 * \note This class is handling in-place tab name editing for open workspaces.
 */
class TabRenamerEventFilter : public QObject
{
  public:
	TabRenamerEventFilter(PVGuiQt::PVOpenWorkspaceTabBar* tab_bar, int index, QLineEdit* line_edit)
	    : _tab_bar(tab_bar), _index(index), _line_edit(line_edit)
	{
	}

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

	/*! \brief Handle the resizing of the tabs for prettier TextElideMode display than QT's way.
	 */
	void resizeEvent(QResizeEvent* event) override;

  protected:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;

	void start_drag(QWidget* workspace);

  protected:
	PVWorkspacesTabWidgetBase* _tab_widget;
	QPoint _drag_start_position;
	bool _drag_ongoing = false;

  private:
	static constexpr int MIN_WIDTH = 160;
	/*!< The manimum width of a tab label in pixel.*/ // size of proxy_sample.log
};

/**
 * \class PVOpenWorkspaceTabBar
 *
 * \note This class is a PVSceneTabBar derivation handling tab bar event for
 *PVOpenWorkspacesTabWidget.
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
	PVWorkspacesTabWidgetBase(Inendi::PVRoot& root, QWidget* parent = 0);

  public:
	/*! \brief Add a workspace with or without animation.
	 */
	virtual int add_workspace(PVWorkspaceBase* page, const QString& label, bool animation = true);

	/*! \brief Remove a workspace with or without animation.
	 */
	virtual void remove_workspace(int index);

	/*! \brief Returns the number of affective tabs in the widget (ie: special tab "+" button is not
	 * taken into account).
	 */
	int count() const { return _tab_bar->count(); }

  protected:
	inline Inendi::PVRoot const& get_root() const { return _root; }
	inline Inendi::PVRoot& get_root() { return _root; }

  signals:
	/*! \brief Signal emitted when a workspace is dragged outside of a PVMainWindow.
	 */
	void workspace_dragged_outside(QWidget*);

  protected slots:
	/*! \brief Slot called when the user closes a workspace.
	 */
	void tab_close_requested(int index);

	/*! \brief Handle the resizing of the tabs for prettier TextElideMode display than QT's way.
	 */
	void resizeEvent(QResizeEvent* event) override;

  private slots:
	/*! \brief Change the CSS property width of the selected tab (used by the animation).
	 */
	void set_tab_width(int tab_width);

	/*! \brief Dummy slot to avoid warning with the tab_width Q_PROPERTY.
	 */
	int get_tab_width() const { return 0; }

	/*! \brief Remove tab at then end of the animation.
	 */
	void animation_state_changed(QAbstractAnimation::State new_state,
	                             QAbstractAnimation::State old_state);

  protected:
	PVSceneTabBar* _tab_bar;

  private:
	int _tab_animated_width;
	bool _tab_animation_ongoing = false;
	int _tab_animation_index;

	Inendi::PVRoot& _root;
};

/**
 * \class PVSceneWorkspacesTabWidget
 *
 * \note This class is a PVWorkspacesTabWidgetBase derivation representing a scene tab widget.
 */
class PVSceneWorkspacesTabWidget : public PVWorkspacesTabWidgetBase
{
	Q_OBJECT
	friend class PVSceneTabBar;

  public:
	PVSceneWorkspacesTabWidget(Inendi::PVScene& scene, QWidget* parent = 0);

  public:
	bool is_project_modified() { return _project_modified; }
	bool is_project_untitled() { return _project_untitled; }

	Inendi::PVScene* get_scene() { return _obs_scene.get_object(); }

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
	/*! \brief Call Inendi::PVRoot::select_source throught the Hive to keep track of current source.
	 */
	void tab_changed(int index);

  private slots:
	void set_project_modified(bool modified = true, QString path = QString());

  private:
	bool _project_modified = false;
	bool _project_untitled = true;

	PVHive::PVObserverSignal<Inendi::PVScene> _obs_scene;
};

/**
 * \class PVOpenWorkspacesTabWidget
 *
 * \note This class is a PVWorkspacesTabWidgetBase derivation representing an open workspace tab
 *widget.
 */
class PVOpenWorkspacesTabWidget : public PVWorkspacesTabWidgetBase
{
	Q_OBJECT
	friend class PVOpenWorkspaceTabBar;

  public:
	PVOpenWorkspacesTabWidget(Inendi::PVRoot& root, QWidget* parent = 0);

  public:
	PVOpenWorkspace* current_workspace() const;
	PVOpenWorkspace* current_workspace_or_create();

  protected:
	/*! \brief Special behavior on workspace creation (handle drag&drop tab switch).
	 */
	void tabInserted(int index) override;

	/*! \brief Special behavior on workspace removal (emit "is_empty" signal when the last source is
	 * closed).
	 */
	void tabRemoved(int index) override;

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
