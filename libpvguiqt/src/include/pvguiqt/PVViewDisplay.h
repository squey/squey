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

#ifndef __PVGUIQT_PVVIEWDISPLAY_H__
#define __PVGUIQT_PVVIEWDISPLAY_H__

#include <QAction>
#include <QCloseEvent>
#include <QContextMenuEvent>
#include <QDockWidget>
#include <QEvent>
#include <QList>
#include <QFocusEvent>
#include <QSignalMapper>

#include <pvbase/types.h>

#include <sigc++/sigc++.h>

#include <functional>

class QString;
class QPoint;
class QWidget;

namespace Inendi
{
class PVView;
} // namespace Inendi

namespace PVGuiQt
{

class PVWorkspaceBase;
class PVSourceWorkspace;

/**
 * \class PVViewDisplay
 *
 * \note This class is a dockable wrapper for graphical view representations.
 */
class PVViewDisplay : public QDockWidget, public sigc::trackable
{
	Q_OBJECT;

	friend PVWorkspaceBase;
	friend PVSourceWorkspace;

	enum EState { HIDDEN, CAN_MAXIMIZE, CAN_RESTORE };

  public:
	/*! \brief Call Inendi::PVRoot::select_view through the Hive.
	 * This is called by the application level events filter DisplaysFocusInEventFilter on
	 * PVViewDisplay QEvent::FocusIn events.
	 */
	void set_current_view();

  public:
	Inendi::PVView* get_view() { return _view; }
	void set_view(Inendi::PVView* view) { _view = view; }

  protected:
	/*! \brief Filter events to allow a PVViewDisplay to be docked inside any other PVWorkspace.
	 */
	bool event(QEvent* event) override;

	/*! \brief Create the view display right click menu.
	 */
	void contextMenuEvent(QContextMenuEvent* event) override;

  private Q_SLOTS:
	/*! \brief Store the state of the drag&drop operation.
	 */
	void drag_started(bool started);

	/*! \brief Store the state of the drag&drop operation.
	 */
	void drag_ended();

	/*! \brief Create the view display right click menu.
	 */
	void plotting_updated(QList<PVCol> const& cols_updated);

	void restore();

	/*! \brief Maximize a view display on a given screen.
	 */
	void maximize_on_screen(int screen_number);

  Q_SIGNALS:
	/*! \brief Signal emited when the display is moved in order to detected a potential tab change.
	 */
	void try_automatic_tab_switch();

  private:
	/*! \brief Register the view to handle several events.
	 */
	void register_view(Inendi::PVView* view);

  private:
	/*! \brief Creates a view display.
	 *  \param[in] view The underlying PVView.
	 *  \param[in] view_widget The widget displayed by the dock widget.
	 *  \param[in] name The function returning the name of the display based on its type.
	 *  \param[in] can_be_central_widget Specifies if the display can be set as central display.
	 *  \param[in] delete_on_close Specifies if the display is deleted when closed.
	 *  \param[in] workspace The parent workspace.
	 *
	 *  \note this constructor is intended to be called only by PVWorkspace, hence the private
	 *visibility.
	 */
	PVViewDisplay(Inendi::PVView* view,
	              QWidget* view_widget,
	              QString name,
	              bool can_be_central_widget,
	              bool delete_on_close,
	              PVWorkspaceBase* parent);

  private:
	Inendi::PVView* _view;
	QString _name;
	PVWorkspaceBase* _workspace;
	QPoint _press_pt;
	bool _can_be_central_widget;

	int _width;
	int _height;
	int _x;
	int _y;
	EState _state = HIDDEN;

	QSignalMapper* _screenSignalMapper;
};
} // namespace PVGuiQt

#endif // #ifndef __PVGUIQT_PVVIEWDISPLAY_H__
