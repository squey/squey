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

#ifndef __PVGUIQT_PVWORKSPACE_H__
#define __PVGUIQT_PVWORKSPACE_H__

class QAction;
class QEvent;
class QToolButton;
class QComboBox;
class QObject;
class QWidget;
#include <QList>

#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVListDisplayDlg.h>

#include <pvdisplays/PVDisplaysContainer.h>
#include <pvdisplays/PVDisplayIf.h>

#include <squey/PVView.h>

namespace Squey
{
class PVView;
class PVSource;
} // namespace Squey

Q_DECLARE_METATYPE(Squey::PVView*)

namespace PVDisplays
{
class PVDisplaySourceIf;
class PVDisplayViewIf;
class PVDisplayViewAxisIf;
class PVDisplayViewZoneIf;
} // namespace PVDisplays

namespace PVGuiQt
{

class PVViewDisplay;

/**
 * \class PVWorkspaceBase
 *
 * \note This class is the base class for workspaces.
 */
class PVWorkspaceBase : public PVDisplays::PVDisplaysContainer
{
	Q_OBJECT

	friend class PVViewDisplay;
	friend class PVViewWidgets;

  private:
	class PVViewWidgets
	{
	  public:
		PVGuiQt::PVAxesCombinationDialog* _axes_combination_editor;
		PVViewWidgets(Squey::PVView* view, PVWorkspaceBase* tab)
		    : _axes_combination_editor(new PVAxesCombinationDialog(*view, tab))
		{
		}
		PVViewWidgets() { _axes_combination_editor = nullptr; }
		~PVViewWidgets(){};

	  protected:
		void delete_widgets() { _axes_combination_editor->deleteLater(); }
	};

  public:
	PVViewWidgets const& get_view_widgets(Squey::PVView* view);
	PVAxesCombinationDialog* get_axes_combination_editor(Squey::PVView* view)
	{
		PVViewWidgets const& widgets = get_view_widgets(view);
		return widgets._axes_combination_editor;
	}

  public:
	explicit PVWorkspaceBase(QWidget* parent) : PVDisplays::PVDisplaysContainer(parent) {}
	~PVWorkspaceBase() override;

  public:
	/*! \brief Create a view display from a widget and add it to the workspace.
	 *
	 *  \param[in] view The underlying PVView.
	 *  \param[in] view_widget The widget displayed by the dock widget.
	 *  \param[in] can_be_central_widget Specifies if the display can be set as central display.
	 *  \param[in] delete_on_close Specifies if the display is deleted when closed.
	 *  \param[in] area The area of the QDockWidget on the QMainWindow.
	 *
	 *  \return A pointer to the view display.
	 */
	PVViewDisplay* add_view_display(Squey::PVView* view,
	                                QWidget* view_display,
	                                PVDisplays::PVDisplayIf& display_if,
	                                bool delete_on_close = true,
	                                Qt::DockWidgetArea area = Qt::TopDockWidgetArea);

	/*! \brief Set a widget as the cental view display of the workspace.
	 *
	 *  \param[in] view The underlying PVView.
	 *  \param[in] view_widget The widget displayed by the dock widget.
	 *  \param[in] delete_on_close Specifies if the display is deleted when closed.
	 *
	 *  \return A pointer to the view display.
	 */
	PVViewDisplay* set_central_display(Squey::PVView* view,
	                                   QWidget* view_widget,
	                                   bool has_help_page,
	                                   bool delete_on_close);

	/*! \brief Create or display the widget used by the view display.
	 *
	 *  \param[in] act The QAction triggering the creation/display of the widget.
	 */
	void toggle_unique_source_widget(QAction* act,
	                                 PVDisplays::PVDisplaySourceIf& display_if,
	                                 Squey::PVSource* src);

	/*! \brief Return the workspace located under the mouse.
	 */
	static PVWorkspaceBase* workspace_under_mouse();

  public:
	static bool drag_started() { return _drag_started; }
	inline int z_order() { return _z_order_index; }

  protected:
	/*! \brief Keep track of the Z Order of the workspace.
	 *
	 *  \note Used by workspace_under_mouse to disambiguate overlapping workspaces.
	 */
	void changeEvent(QEvent* event) override;

  protected:
  	/*! \brief Forward mouse buttons legend changed signals to the status bar
	 */
  	void track_mouse_buttons_legend_changed(PVDisplays::PVDisplayIf& display_if, QWidget* widget);

  public Q_SLOTS:
	/*! \brief Create the widget used by the view display.
	 *
	 *  \param[in] act The QAction triggering the creation of the widget.
	 */
	void create_view_widget(PVDisplays::PVDisplayViewIf& iface,
	                        Squey::PVView* view,
	                        std::vector<std::any> params = {}) override;

  private Q_SLOTS:
	/*! \brief Switch a view display with the central widget.
	 *
	 *  \param[in] display_dock The view display to switch as central widget.
	 *
	 *  \note In order to keep the displays positions, the displays themselves are
	 *        not really switched: instead we switch all of their content (name, colors,
	 *observers...)
	 */
	void switch_with_central_widget(PVViewDisplay* display_dock = nullptr);

	/*! \brief Update the internal list of displays when one of them are destroyed.
	 *
	 *  \param[in] object The destroyed view display.
	 *
	 *  \note The internal list of displays is used by toggle_unique_source_widget.
	 */
	void display_destroyed(QObject* object = 0);

  Q_SIGNALS:
	/*! \brief Signal forwarded when a display is moved in order to detected a potential tab change.
	 */
	void try_automatic_tab_switch();

  protected:
	QList<PVViewDisplay*> _displays;
	int _z_order_index = 0;
	static uint64_t _z_order_counter;
	static bool _drag_started;
	QHash<Squey::PVView const*, PVViewWidgets> _view_widgets;
};

/**
 * \class PVSourceWorkspace
 *
 * \note This class is a PVWorkspaceBase derivation representing workspaces related to a source.
 */
class PVSourceWorkspace : public PVWorkspaceBase
{
	Q_OBJECT

  public:
	template <class T>
	using list_display = QList<std::pair<QToolButton*, T*>>;

  public:
	explicit PVSourceWorkspace(Squey::PVSource* source, QWidget* parent = nullptr);

  public:
	inline Squey::PVSource* get_source() const { return _source; }

	bool has_errors_or_warnings() const;
	QString source_type() const;

	/**
	 * Get the Dialog widget that show invalid elements.
	 */
	inline PVGuiQt::PVListDisplayDlg* get_source_invalid_evts_dlg() const { return _inv_evts_dlg; }

	template <class T>
	void populate_display();

  private:
	Squey::PVSource* _source = nullptr;
	QToolBar* _toolbar = nullptr;
	QComboBox* _toolbar_combo_views = nullptr;

	PVGuiQt::PVListDisplayDlg* _inv_evts_dlg = nullptr; //<! Dialog with listing of invalid elements.
};
} // namespace PVGuiQt

#endif /* __PVGUIQT_PVWORKSPACE_H__ */
