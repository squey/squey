/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVGUIQT_PVWORKSPACESTABWIDGET_H__
#define __PVGUIQT_PVWORKSPACESTABWIDGET_H__

#include <sigc++/sigc++.h>

#include <squey/PVScene.h>

#include <QPoint>
#include <QTabBar>
#include <QTabWidget>

class QMouseEvent;
class QWidget;
class QStackedWidget;

namespace PVGuiQt
{

class PVWorkspaceBase;
class PVSceneWorkspacesTabWidget;
class PVImportWorkflowTabBar;

/**
 * \class PVSceneTabBar
 *
 * \note This class is handling tab bar event for PVSceneWorkspacesTabWidget.
 */
class PVSceneTabBar : public QTabBar
{
	Q_OBJECT

  public:
	explicit PVSceneTabBar(QWidget* parent = nullptr);

  public:
	/*! \brief Handle the resizing of the tabs for prettier TextElideMode display than QT's way.
	 */
	//void resizeEvent(QResizeEvent* event) override;

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
class PVSceneWorkspacesTabWidget : public QWidget, public sigc::trackable
{
	Q_OBJECT
	Q_PROPERTY(int tab_width READ get_tab_width WRITE set_tab_width);

  private:
	enum class EImportWorkflowStage : size_t {
		//DATA = 0,
		FORMAT = 0,
		ERRORS,
		WORKSPACE
	};

  public:
	explicit PVSceneWorkspacesTabWidget(Squey::PVScene& scene, QWidget* parent = 0);

  public:
	/*! \brief Add a workspace with or without animation.
	 */
	void add_workspace(PVWorkspaceBase* page, const QString& label);

	void show_errors_and_warnings();

	/*! \brief Remove a workspace with or without animation.
	 */
	void remove_workspace(int index);

	int current_index() const { return _workspace_tab_bar->currentIndex(); }

	void set_current_tab(int index);
	void set_current_workflow_tab(int index);

	void set_worflow_tab_status(int index);

	QWidget* current_widget();

	int index_of(QWidget* workspace);

	QString tab_text(int index) { return _workspace_tab_bar->tabText(index); }

  public:
	bool is_project_modified() { return _project_modified; }

	Squey::PVScene& get_scene() { return _scene; }

  public Q_SLOTS:
	/*! \brief Call Squey::PVRoot::select_source to keep track of current source.
	 */
	void tab_changed(int index);

  protected Q_SLOTS:
	/*! \brief Slot called when the user closes a workspace.
	 */
	void tab_close_requested(int index);

	// /*! \brief Handle the resizing of the tabs for prettier TextElideMode display than QT's way.
	//  */
	// void resizeEvent(QResizeEvent* event) override;

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
	Squey::PVScene& _scene;
	PVGuiQt::PVSceneTabBar* _workspace_tab_bar = nullptr;
	PVGuiQt::PVImportWorkflowTabBar* _import_worflow_tab_bar = nullptr;
	QStackedWidget* _stacked_widget = nullptr;
	QStackedWidget* _stacked_widget_data = nullptr;
	QStackedWidget* _stacked_widget_format = nullptr;
	QStackedWidget* _stacked_widget_errors = nullptr;
	QStackedWidget* _stacked_widget_workspace = nullptr;
	bool _project_modified = false;
};
} // namespace PVGuiQt

#endif // __PVGUIQT_PVWORKSPACESTABWIDGET_H__
