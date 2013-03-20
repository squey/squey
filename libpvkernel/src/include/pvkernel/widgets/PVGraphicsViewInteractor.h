#ifndef PVWIDGETS_PVGRAPHICSVIEWINTERACTOR_H
#define PVWIDGETS_PVGRAPHICSVIEWINTERACTOR_H

#include <QObject>
#include <QWheelEvent>
#include <QKeyEvent>

#include <type_traits>

namespace PVWidgets {

class PVGraphicsView;

class PVGraphicsViewInteractorBase: public QObject
{
	Q_OBJECT

	friend class PVGraphicsView;

protected:
	PVGraphicsViewInteractorBase(PVGraphicsView* parent);
};

/*! \brief Template class that defines an interactor for a PVGraphicsView-based class.
 *
 * This class defines an interactor for a PVGraphicsView-based class. Thus, T must be a subclass of PVGraphicsView.
 * It is recommanded to set a public typedef for any PVGraphicsView-based class, so that it is easier to create a new interactor for that type of view.
 *
 * For instance, in the parallel view library, this typedef is defined in PVZoomableDrawingAreaInteractor.h:
 *
 * \code
 * typedef PVWidgets::PVGraphicsViewInteractor<PVZoomableDrawingArea> PVZoomableDrawingAreaInteractor;
 * \endcode
 */
template <class T>
class PVGraphicsViewInteractor: public PVGraphicsViewInteractorBase
{
	typedef T object_type;
	friend class PVGraphicsView;

protected:
	PVGraphicsViewInteractor(PVGraphicsView* parent):
		PVGraphicsViewInteractorBase(parent)
	{}

	virtual	~PVGraphicsViewInteractor()
	{}

protected:
	/*! \brief Called when a mouse button press event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QMouseEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool mousePressEvent(object_type* obj, QMouseEvent* event) { return false; }

	/*! \brief Called when a mouse button release event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QMouseEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool mouseReleaseEvent(object_type* obj, QMouseEvent* event) { return false; }

	/*! \brief Called when a mouse move event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QMouseEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool mouseMoveEvent(object_type* obj, QMouseEvent* event) { return false; }

	/*! \brief Called when a wheel event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QWheelEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool wheelEvent(object_type* obj, QWheelEvent* event) { return false; }

	/*! \brief Called when a key event has occured.
	 *  \param[in] obj   A pointer to the view that received the event
	 *  \param[in] event a QKeyEvent object that describes the event
	 *
	 *  \return true if the event has been processed and must not be processed
	 *  by the other interactors. false otherwise.
	 */
	virtual bool keyPressEvent(object_type* obj, QKeyEvent* event) { return false; }

private:
	bool eventFilter(QObject* obj, QEvent* event) override
	{
		object_type* real_obj = qobject_cast<object_type*>(obj);
		if (!real_obj) {
			return QObject::eventFilter(obj, event);
		}

		switch (event->type())
		{
			case QEvent::Wheel:
				return wheelEvent(real_obj, static_cast<QWheelEvent*>(event));
			case QEvent::MouseButtonPress:
				return mousePressEvent(real_obj, static_cast<QMouseEvent*>(event));
			case QEvent::MouseButtonRelease:
				return mouseReleaseEvent(real_obj, static_cast<QMouseEvent*>(event));
			case QEvent::MouseMove:
				return mouseMoveEvent(real_obj, static_cast<QMouseEvent*>(event));
			case QEvent::KeyPress:
				return keyPressEvent(real_obj, static_cast<QKeyEvent*>(event));
			default:
				break;
		}

		return QObject::eventFilter(obj, event);
	}
};

}

#endif
