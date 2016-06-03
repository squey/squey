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

class PVWorkspaceBase;
class PVSceneWorkspacesTabWidget;
class PVSceneWorkspacesTabWidget;
class PVSceneTabBar;

/**
 * \class PVSceneTabBar
 *
 * \note This class is handling tab bar event for PVSceneWorkspacesTabWidget.
 */
class PVSceneTabBar : public QTabBar
{
	Q_OBJECT

  public:
	PVSceneTabBar(PVSceneWorkspacesTabWidget* tab_widget);

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
class PVSceneWorkspacesTabWidget : public QTabWidget
{
	Q_OBJECT
	Q_PROPERTY(int tab_width READ get_tab_width WRITE set_tab_width);

  public:
	PVSceneWorkspacesTabWidget(Inendi::PVScene& scene, QWidget* parent = 0);

  public:
	/*! \brief Add a workspace with or without animation.
	 */
	void add_workspace(PVWorkspaceBase* page, const QString& label);

	/*! \brief Remove a workspace with or without animation.
	 */
	void remove_workspace(int index);

  public:
	bool is_project_modified() { return _project_modified; }
	bool is_project_untitled() { return _project_untitled; }

	Inendi::PVScene* get_scene() { return _obs_scene.get_object(); }

  protected:
	inline Inendi::PVRoot const& get_root() const { return _root; }
	inline Inendi::PVRoot& get_root() { return _root; }

  public slots:
	/*! \brief Call Inendi::PVRoot::select_source throught the Hive to keep track of current source.
	 */
	void tab_changed(int index);

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
	int get_tab_width() const
	{
		// It will be called for initilize tab_width value before we set it to Start value of the
		// animation
		return 0;
	}

	void animation_finished();

	void set_project_modified(bool modified = true, QString path = QString());

  signals:
	void project_modified(bool, QString = QString());

	/*! \brief Signal emitted when the last source is closed.
	 */
	void is_empty();

	/*! \brief Signal emitted when a workspace is dragged outside of a PVMainWindow.
	 */
	void workspace_dragged_outside(QWidget*);

  private:
	Inendi::PVRoot& _root;

	bool _project_modified = false;
	bool _project_untitled = true;

	PVHive::PVObserverSignal<Inendi::PVScene> _obs_scene;
};
}

#endif // __PVGUIQT_PVWORKSPACESTABWIDGET_H__
