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

template <class T>
class PVGraphicsViewInteractor: public PVGraphicsViewInteractorBase
{
	typedef T object_type;
	friend class PVGraphicsView;

protected:
	PVGraphicsViewInteractor(PVGraphicsView* parent):
		PVGraphicsViewInteractorBase(parent)
	{ }

protected:
	virtual bool wheelEvent(object_type* obj, QWheelEvent* event) { return false; }
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
