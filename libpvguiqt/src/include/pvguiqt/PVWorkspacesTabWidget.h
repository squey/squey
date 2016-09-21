/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVGUIQT_PVWORKSPACESTABWIDGET_H__
#define __PVGUIQT_PVWORKSPACESTABWIDGET_H__

#include <sigc++/sigc++.h>

#include <inendi/PVScene.h>

#include <QPoint>
#include <QTabBar>
#include <QTabWidget>

class QMouseEvent;
class QWidget;

namespace PVGuiQt
{

class PVWorkspaceBase;
class PVSceneWorkspacesTabWidget;

/**
 * \class PVSceneTabBar
 *
 * \note This class is handling tab bar event for PVSceneWorkspacesTabWidget.
 */
class PVSceneTabBar : public QTabBar
{
	Q_OBJECT

  public:
	explicit PVSceneTabBar(PVSceneWorkspacesTabWidget* tab_widget);

  public:
	/*! \brief Handle the resizing of the tabs for prettier TextElideMode display than QT's way.
	 */
	void resizeEvent(QResizeEvent* event) override;

  protected:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;

	void start_drag(QWidget* workspace);

  protected:
	PVSceneWorkspacesTabWidget* _tab_widget;
	QPoint _drag_start_position;
	bool _drag_ongoing = false;

  private:
	// size of proxy_sample.log
	static constexpr int MIN_WIDTH = 160; //!< The manimum width of a tab label in pixel.
};

/**
 * \class PVSceneWorkspacesTabWidget
 *
 * \note This class is a PVSceneWorkspacesTabWidget derivation representing a scene tab widget.
 * ie: It is the tab widget with all sources of the scene and tab modification add/updage/change
 * sources.
 */
class PVSceneWorkspacesTabWidget : public QTabWidget, public sigc::trackable
{
	Q_OBJECT
	Q_PROPERTY(int tab_width READ get_tab_width WRITE set_tab_width);

  public:
	explicit PVSceneWorkspacesTabWidget(Inendi::PVScene& scene, QWidget* parent = 0);

  public:
	/*! \brief Add a workspace with or without animation.
	 */
	void add_workspace(PVWorkspaceBase* page, const QString& label);

	/*! \brief Remove a workspace with or without animation.
	 */
	void remove_workspace(int index);

  public:
	bool is_project_modified() { return _project_modified; }

	Inendi::PVScene& get_scene() { return _scene; }

  public Q_SLOTS:
	/*! \brief Call Inendi::PVRoot::select_source to keep track of current source.
	 */
	void tab_changed(int index);

  protected Q_SLOTS:
	/*! \brief Slot called when the user closes a workspace.
	 */
	void tab_close_requested(int index);

	/*! \brief Handle the resizing of the tabs for prettier TextElideMode display than QT's way.
	 */
	void resizeEvent(QResizeEvent* event) override;

  private Q_SLOTS:
	/*! \brief Change the CSS property width of the selected tab (used by the animation).
	 */
	void set_tab_width(int tab_width);

	/*! \brief Dummy slot to avoid warning with the tab_width Q_PROPERTY.
	 */
	int get_tab_width() const
	{
		// It will be called for initilize tab_width value before we set it to Start value of the
		// animation
		return 0;
	}

	void animation_finished();

	void set_project_modified();

  Q_SIGNALS:
	void project_modified();

	/*! \brief Signal emitted when the last source is closed.
	 */
	void is_empty();

	/*! \brief Signal emitted when a workspace is dragged outside of a PVMainWindow.
	 */
	void workspace_dragged_outside(QWidget*);

  private:
	Inendi::PVScene& _scene;
	bool _project_modified = false;
};
}

#endif // __PVGUIQT_PVWORKSPACESTABWIDGET_H__
