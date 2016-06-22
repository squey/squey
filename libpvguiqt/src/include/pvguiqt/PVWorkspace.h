/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVGUIQT_PVWORKSPACE_H__
#define __PVGUIQT_PVWORKSPACE_H__

class QAction;
class QEvent;
class QToolButton;
class QObject;
class QWidget;
#include <QList>

#include <pvguiqt/PVAxesCombinationDialog.h>
#include <pvguiqt/PVListDisplayDlg.h>

#include <pvdisplays/PVDisplaysContainer.h>

#include <inendi/PVView.h>

/**
 * This helper function is std::get with type as parameter available only from C++14
 */
template <class T, size_t I, class... U>
typename std::enable_if<
    std::is_same<typename std::tuple_element<I, std::tuple<U...>>::type, T>::value,
    T&>::type
__get_typed_arg(std::tuple<U...>& t)
{
	return std::get<I>(t);
}

template <class T, size_t I, class... U>
typename std::enable_if<
    not std::is_same<typename std::tuple_element<I, std::tuple<U...>>::type, T>::value,
    T&>::type
__get_typed_arg(std::tuple<U...>& t)
{
	return __get_typed_arg<T, I - 1>(t);
}

template <class T, class... U>
T& get_typed_arg(std::tuple<U...>& u)
{
	return __get_typed_arg<T, sizeof...(U)-1>(u);
}

namespace Inendi
{
class PVView;
class PVSource;
}

Q_DECLARE_METATYPE(Inendi::PVView*)

namespace PVDisplays
{
class PVDisplayViewIf;
class PVDisplayViewAxisIf;
class PVDisplayViewZoneIf;
}

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
		PVViewWidgets(Inendi::PVView* view, PVWorkspaceBase* tab)
		    : _axes_combination_editor(new PVAxesCombinationDialog(*view, tab))
		{
		}
		PVViewWidgets() { _axes_combination_editor = nullptr; }
		~PVViewWidgets(){};

	  protected:
		void delete_widgets() { _axes_combination_editor->deleteLater(); }
	};

  public:
	PVViewWidgets const& get_view_widgets(Inendi::PVView* view);
	PVAxesCombinationDialog* get_axes_combination_editor(Inendi::PVView* view)
	{
		PVViewWidgets const& widgets = get_view_widgets(view);
		return widgets._axes_combination_editor;
	}

  public:
	PVWorkspaceBase(QWidget* parent) : PVDisplays::PVDisplaysContainer(parent) {}
	virtual ~PVWorkspaceBase();

  public:
	/*! \brief Create a view display from a widget and add it to the workspace.
	 *
	 *  \param[in] view The underlying PVView.
	 *  \param[in] view_widget The widget displayed by the dock widget.
	 *  \param[in] name The function returning the name of the display based on its type.
	 *  \param[in] can_be_central_widget Specifies if the display can be set as central display.
	 *  \param[in] delete_on_close Specifies if the display is deleted when closed.
	 *  \param[in] area The area of the QDockWidget on the QMainWindow.
	 *
	 *  \return A pointer to the view display.
	 */
	PVViewDisplay* add_view_display(Inendi::PVView* view,
	                                QWidget* view_display,
	                                std::function<QString()> name,
	                                bool can_be_central_display = true,
	                                bool delete_on_close = true,
	                                Qt::DockWidgetArea area = Qt::TopDockWidgetArea);

	/*! \brief Set a widget as the cental view display of the workspace.
	 *
	 *  \param[in] view The underlying PVView.
	 *  \param[in] view_widget The widget displayed by the dock widget.
	 *  \param[in] name The function returning the name of the display based on its type.
	 *  \param[in] delete_on_close Specifies if the display is deleted when closed.
	 *
	 *  \return A pointer to the view display.
	 */
	PVViewDisplay* set_central_display(Inendi::PVView* view,
	                                   QWidget* view_widget,
	                                   std::function<QString()> name,
	                                   bool delete_on_close);

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

  public Q_SLOTS:
	/*! \brief Create the widget used by the view display.
	 *
	 *  \param[in] act The QAction triggering the creation of the widget.
	 */
	void create_view_widget(QAction* act = nullptr) override;

  public:
	template <class T>
	struct Tag {
	};

	template <class T>
	void create_view_dispatch(QAction*, Tag<T>);

	void create_view_dispatch(QAction* act, Tag<PVDisplays::PVDisplayViewIf>)
	{
		create_view_widget(act);
	}
	void create_view_dispatch(QAction* act, Tag<PVDisplays::PVDisplayViewAxisIf>)
	{
		create_view_axis_widget(act);
	}
	void create_view_dispatch(QAction* act, Tag<PVDisplays::PVDisplayViewZoneIf>)
	{
		create_view_zone_widget(act);
	}

  private Q_SLOTS:
	/*! \brief Create the widget used by the view display with axis parameter.
	 *
	 *  \param[in] act The QAction triggering the creation of the widget.
	 */
	void create_view_axis_widget(QAction* act = nullptr) override;

	/*! \brief Create the widget used by the view display with zone parameter.
	 *
	 *  \param[in] act The QAction triggering the creation of the widget.
	 */
	void create_view_zone_widget(QAction* act = nullptr) override;

	/*! \brief Create or display the widget used by the view display.
	 *
	 *  \param[in] act The QAction triggering the creation/display of the widget.
	 */
	void toggle_unique_source_widget(QAction* act = nullptr) override;

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
	QHash<Inendi::PVView const*, PVViewWidgets> _view_widgets;
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
	PVSourceWorkspace(Inendi::PVSource* source, QWidget* parent = nullptr);

  public:
	inline Inendi::PVSource* get_source() const { return _source; }

	/**
	 * Get the Dialog widget that show invalid elements.
	 */
	inline PVGuiQt::PVListDisplayDlg* get_source_invalid_evts_dlg() const { return _inv_evts_dlg; }

	template <class T>
	void fill_display();

	template <class T>
	void populate_display();

  private:
	Inendi::PVSource* _source = nullptr;
	QToolBar* _toolbar;

	PVGuiQt::PVListDisplayDlg* _inv_evts_dlg; //<! Dialog with listing of invalid elements.

	std::tuple<typename PVSourceWorkspace::list_display<PVDisplays::PVDisplayViewIf>,
	           typename PVSourceWorkspace::list_display<PVDisplays::PVDisplayViewAxisIf>,
	           typename PVSourceWorkspace::list_display<PVDisplays::PVDisplayViewZoneIf>>
	    _tool_buttons;
};
}

#endif /* __PVGUIQT_PVWORKSPACE_H__ */
