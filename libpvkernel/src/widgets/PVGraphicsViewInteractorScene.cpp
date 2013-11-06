
#include <pvkernel/widgets/PVGraphicsViewInteractorScene.h>

#include <pvkernel/widgets/PVGraphicsView.h>

#include <QGraphicsSceneWheelEvent>
#include <QApplication>
#include <QGraphicsScene>

template <typename E>
static inline void propagate_event_to_scene(PVWidgets::PVGraphicsView* obj, E* event)
{
	if (obj->get_scene()) {
		QApplication::sendEvent(obj->get_scene(), event);
	}
}

/*****************************************************************************
 * PVWidgets::PVGraphicsViewInteractorScene::PVGraphicsViewInteractorScene
 *****************************************************************************/

PVWidgets::PVGraphicsViewInteractorScene::PVGraphicsViewInteractorScene(PVGraphicsView* parent) :
	PVWidgets::PVGraphicsViewInteractor<PVGraphicsView>(parent)
{}

/*****************************************************************************
 * PVWidgets::PVGraphicsViewInteractorScene::contextMenuEvent
 *****************************************************************************/

bool PVWidgets::PVGraphicsViewInteractorScene::contextMenuEvent(PVGraphicsView* obj,
                                                                QContextMenuEvent* event)
{
	obj->_mouse_pressed_screen_coord = event->globalPos();
	obj->_mouse_pressed_view_coord = event->pos();
	obj->_mouse_pressed_scene_coord = obj->map_to_scene(obj->_mouse_pressed_view_coord);

	obj->_last_mouse_move_screen_coord = obj->_mouse_pressed_screen_coord;
	obj->_last_mouse_move_scene_coord = obj->_mouse_pressed_scene_coord;

	QGraphicsSceneContextMenuEvent scene_event(QEvent::GraphicsSceneContextMenu);

	scene_event.setWidget(obj->get_viewport());

	scene_event.setReason((QGraphicsSceneContextMenuEvent::Reason)(event->reason()));

	scene_event.setScenePos(obj->_mouse_pressed_scene_coord);
	scene_event.setScreenPos(obj->_mouse_pressed_screen_coord);

	propagate_event_to_scene(obj, &scene_event);

	if (scene_event.isAccepted()) {
		event->setAccepted(true);
		return true;
	}

	return false;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsViewInteractorScene::mouseDoubleClickEvent
 *****************************************************************************/

bool PVWidgets::PVGraphicsViewInteractorScene::mouseDoubleClickEvent(PVGraphicsView* obj,
                                                                     QMouseEvent* event)
{
	obj->_mouse_pressed_button = event->button();

	obj->_mouse_pressed_screen_coord = event->globalPos();
	obj->_mouse_pressed_view_coord = event->pos();
	obj->_mouse_pressed_scene_coord = obj->map_to_scene(obj->_mouse_pressed_view_coord);

	obj->_last_mouse_move_screen_coord = obj->_mouse_pressed_screen_coord;
	obj->_last_mouse_move_scene_coord = obj->_mouse_pressed_scene_coord;

	QGraphicsSceneMouseEvent scene_event(QEvent::GraphicsSceneMouseDoubleClick);

	scene_event.setWidget(obj->get_viewport());

	scene_event.setButtonDownScenePos(obj->_mouse_pressed_button,
	                                  obj->_mouse_pressed_scene_coord);
	scene_event.setButtonDownScreenPos(obj->_mouse_pressed_button,
	                                   obj->_mouse_pressed_screen_coord);

	scene_event.setScenePos(obj->_mouse_pressed_scene_coord);
	scene_event.setScreenPos(obj->_mouse_pressed_screen_coord);
	scene_event.setLastScenePos(obj->_last_mouse_move_scene_coord);
	scene_event.setLastScreenPos(obj->_last_mouse_move_screen_coord);

	scene_event.setButtons(event->buttons());
	scene_event.setButton(event->button());
	scene_event.setModifiers(event->modifiers());
	scene_event.setAccepted(false);

	propagate_event_to_scene(obj, &scene_event);

	if (scene_event.isAccepted()) {
		event->setAccepted(true);
		return true;
	}

	return false;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsViewInteractorScene::mousePressEvent
 *****************************************************************************/

bool PVWidgets::PVGraphicsViewInteractorScene::mousePressEvent(PVGraphicsView* obj,
                                                               QMouseEvent* event)
{
	obj->_mouse_pressed_button = event->button();

	obj->_mouse_pressed_screen_coord = event->globalPos();
	obj->_mouse_pressed_view_coord = event->pos();
	obj->_mouse_pressed_scene_coord = obj->map_to_scene(obj->_mouse_pressed_view_coord);

	obj->_last_mouse_move_screen_coord = obj->_mouse_pressed_screen_coord;
	obj->_last_mouse_move_scene_coord = obj->_mouse_pressed_scene_coord;

	QGraphicsSceneMouseEvent scene_event(QEvent::GraphicsSceneMousePress);

	scene_event.setWidget(obj->get_viewport());

	scene_event.setButtonDownScenePos(obj->_mouse_pressed_button,
	                                  obj->_mouse_pressed_scene_coord);
	scene_event.setButtonDownScreenPos(obj->_mouse_pressed_button,
	                                   obj->_mouse_pressed_screen_coord);

	scene_event.setScenePos(obj->_mouse_pressed_scene_coord);
	scene_event.setScreenPos(obj->_mouse_pressed_screen_coord);
	scene_event.setLastScenePos(obj->_last_mouse_move_scene_coord);
	scene_event.setLastScreenPos(obj->_last_mouse_move_screen_coord);

	scene_event.setButtons(event->buttons());
	scene_event.setButton(event->button());
	scene_event.setModifiers(event->modifiers());
	scene_event.setAccepted(false);

	propagate_event_to_scene(obj, &scene_event);

	if (scene_event.isAccepted()) {
		event->setAccepted(true);
		return true;
	}

	return false;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsViewInteractorScene::mouseReleaseEvent
 *****************************************************************************/

bool PVWidgets::PVGraphicsViewInteractorScene::mouseReleaseEvent(PVGraphicsView* obj,
                                                                 QMouseEvent* event)
{
	QGraphicsSceneMouseEvent scene_event(QEvent::GraphicsSceneMouseRelease);

	scene_event.setWidget(obj->get_viewport());

	scene_event.setButtonDownScenePos(obj->_mouse_pressed_button, obj->_mouse_pressed_scene_coord);
	scene_event.setButtonDownScreenPos(obj->_mouse_pressed_button, obj->_mouse_pressed_screen_coord);

	scene_event.setScenePos(obj->map_to_scene(event->pos()));
	scene_event.setScreenPos(event->globalPos());
	scene_event.setLastScenePos(obj->_last_mouse_move_scene_coord);
	scene_event.setLastScreenPos(obj->_last_mouse_move_screen_coord);

	scene_event.setButtons(event->buttons());
	scene_event.setButton(event->button());
	scene_event.setModifiers(event->modifiers());

	scene_event.setAccepted(false);

	propagate_event_to_scene(obj, &scene_event);

	if (scene_event.isAccepted()) {
		event->setAccepted(true);
		return true;
	}

	return false;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsViewInteractorScene::mouseMoveEvent
 *****************************************************************************/

bool PVWidgets::PVGraphicsViewInteractorScene::mouseMoveEvent(PVGraphicsView* obj,
                                                              QMouseEvent* event)
{
	QGraphicsSceneMouseEvent scene_event(QEvent::GraphicsSceneMouseMove);

	scene_event.setWidget(obj->get_viewport());

	scene_event.setButtonDownScenePos(obj->_mouse_pressed_button,
	                                  obj->_mouse_pressed_scene_coord);
	scene_event.setButtonDownScreenPos(obj->_mouse_pressed_button,
	                                   obj->_mouse_pressed_screen_coord);

	scene_event.setScenePos(obj->map_to_scene(event->pos()));
	scene_event.setScreenPos(event->globalPos());
	scene_event.setLastScenePos(obj->_last_mouse_move_scene_coord);
	scene_event.setLastScreenPos(obj->_last_mouse_move_screen_coord);

	obj->_last_mouse_move_scene_coord = scene_event.scenePos();
	obj->_last_mouse_move_screen_coord = scene_event.screenPos();

	scene_event.setButtons(event->buttons());
	scene_event.setButton(event->button());
	scene_event.setModifiers(event->modifiers());
	scene_event.setAccepted(false);

	propagate_event_to_scene(obj, &scene_event);

	if (scene_event.isAccepted()) {
		event->setAccepted(true);
		return true;
	}

	return false;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsViewInteractorScene::wheelEvent
 *****************************************************************************/

bool PVWidgets::PVGraphicsViewInteractorScene::wheelEvent(PVGraphicsView* obj,
                                                          QWheelEvent* event)
{
	QGraphicsSceneWheelEvent scene_event(QEvent::GraphicsSceneWheel);

	scene_event.setWidget(obj->get_viewport());

	scene_event.setScenePos(obj->map_to_scene(event->pos()));
	scene_event.setScreenPos(event->globalPos());

	scene_event.setButtons(event->buttons());
	scene_event.setModifiers(event->modifiers());
	scene_event.setDelta(event->delta());
	scene_event.setOrientation(event->orientation());
	scene_event.setAccepted(false);

	propagate_event_to_scene(obj, &scene_event);

	if (scene_event.isAccepted()) {
		event->setAccepted(true);
		return true;
	}
	return false;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsViewInteractorScene::keyPressEvent
 *****************************************************************************/

bool PVWidgets::PVGraphicsViewInteractorScene::keyPressEvent(PVGraphicsView* obj,
                                                             QKeyEvent* event)
{
	propagate_event_to_scene(obj, event);

	if (event->isAccepted()) {
		return true;
	}

	return false;
}

/*****************************************************************************
 * PVWidgets::PVGraphicsViewInteractorScene::keyReleaseEvent
 *****************************************************************************/

bool PVWidgets::PVGraphicsViewInteractorScene::keyReleaseEvent(PVGraphicsView* obj,
                                                               QKeyEvent* event)
{
	propagate_event_to_scene(obj, event);

	if (event->isAccepted()) {
		return true;
	}

	return false;
}
