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

#ifndef PVWIDGETS_PVGRAPHICSVIEWINTERACTOR_H
#define PVWIDGETS_PVGRAPHICSVIEWINTERACTOR_H

#include <QObject>
#include <QWheelEvent>
#include <QKeyEvent>

#include <type_traits>

namespace PVWidgets
{

class PVGraphicsView;

class PVGraphicsViewInteractorBase : public QObject
{
	Q_OBJECT

	friend class PVGraphicsView;

	virtual bool call(QObject* obj, QEvent* event) = 0;

  protected:
	explicit PVGraphicsViewInteractorBase(PVGraphicsView* parent);
};

/*! \brief Template class that defines an interactor for a PVGraphicsView-based class.
 *
 * This class defines an interactor for a PVGraphicsView-based class. Thus, T must be a subclass of
 *PVGraphicsView.
 * It is recommanded to set a public typedef for any PVGraphicsView-based class, so that it is
 *easier to create a new interactor for that type of view.
 *
 * For instance, in the parallel view library, this typedef is defined in
 *PVZoomableDrawingAreaInteractor.h:
 *
 * \code
 * typedef PVWidgets::PVGraphicsViewInteractor<PVZoomableDrawingArea>
 *PVZoomableDrawingAreaInteractor;
 * \endcode
 */
template <class T>
class PVGraphicsViewInteractor : public PVGraphicsViewInteractorBase
{
	typedef T object_type;
	friend class PVGraphicsView;

  protected:
	explicit PVGraphicsViewInteractor(PVGraphicsView* parent) : PVGraphicsViewInteractorBase(parent)
	{
	}

	~PVGraphicsViewInteractor() override = default;

  protected:
	/*! \brief Called when a cotnext menu event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QContextMenuEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool contextMenuEvent(object_type* /*obj*/, QContextMenuEvent* /*event*/)
	{
		return false;
	}

	/*! \brief Called when a mouse button press event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QMouseEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool mouseDoubleClickEvent(object_type* /*obj*/, QMouseEvent* /*event*/)
	{
		return false;
	}

	/*! \brief Called when a mouse button press event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QMouseEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool mousePressEvent(object_type* /*obj*/, QMouseEvent* /*event*/) { return false; }

	/*! \brief Called when a mouse button release event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QMouseEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool mouseReleaseEvent(object_type* /*obj*/, QMouseEvent* /*event*/) { return false; }

	/*! \brief Called when a mouse move event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QMouseEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool mouseMoveEvent(object_type* /*obj*/, QMouseEvent* /*event*/) { return false; }

	/*! \brief Called when a wheel event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QWheelEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool wheelEvent(object_type* /*obj*/, QWheelEvent* /*event*/) { return false; }

	/*! \brief Called when a key event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QKeyEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool keyPressEvent(object_type* /*obj*/, QKeyEvent* /*event*/) { return false; }

	/*! \brief Called when a key event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QKeyEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool keyReleaseEvent(object_type* /*obj*/, QKeyEvent* /*event*/) { return false; }

	/*! \brief Called when a resize event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QResizeEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool resizeEvent(object_type* /*obj*/, QResizeEvent* /*event*/) { return false; }

  protected:
	bool call(QObject* obj, QEvent* event) override
	{
		object_type* real_obj = qobject_cast<object_type*>(obj);
		if (!real_obj) {
			return false;
		}

		switch (event->type()) {
		case QEvent::ContextMenu:
			return contextMenuEvent(real_obj, static_cast<QContextMenuEvent*>(event));
		case QEvent::Wheel:
			return wheelEvent(real_obj, static_cast<QWheelEvent*>(event));
		case QEvent::MouseButtonDblClick:
			return mouseDoubleClickEvent(real_obj, static_cast<QMouseEvent*>(event));
		case QEvent::MouseButtonPress:
			return mousePressEvent(real_obj, static_cast<QMouseEvent*>(event));
		case QEvent::MouseButtonRelease:
			return mouseReleaseEvent(real_obj, static_cast<QMouseEvent*>(event));
		case QEvent::MouseMove:
			return mouseMoveEvent(real_obj, static_cast<QMouseEvent*>(event));
		case QEvent::KeyPress:
			return keyPressEvent(real_obj, static_cast<QKeyEvent*>(event));
		case QEvent::KeyRelease:
			return keyReleaseEvent(real_obj, static_cast<QKeyEvent*>(event));
		case QEvent::Resize:
			return resizeEvent(real_obj, static_cast<QResizeEvent*>(event));
		default:
			break;
		}

		return false;
	}
};
} // namespace PVWidgets

#endif
