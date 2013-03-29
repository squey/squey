/**
 * \file PVParallelScene.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <picviz/PVStateMachine.h>
#include <picviz/PVView.h>

#include <picviz/widgets/editors/PVAxisIndexEditor.h>

#include <pvhive/PVCallHelper.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVSlidersGroup.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoneRendering.h>

#include <tbb/task.h>

#include <QtCore>
#include <QKeyEvent>

#include <QApplication>
#include <QDialog>
#include <QLayout>
#include <QLabel>
#include <QDialogButtonBox>
#include <QRect>

#define CRAND() (127 + (random() & 0x7F))

#define SCENE_MARGIN 32
#define RENDER_TIMER_TIMEOUT 100 // in ms

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::PVFullParallelScene
 *
 *****************************************************************************/
PVParallelView::PVFullParallelScene::PVFullParallelScene(PVFullParallelView* full_parallel_view, Picviz::PVView_sp& view_sp, PVParallelView::PVSlidersManager_p sm_p, PVBCIDrawingBackend& backend, PVZonesManager const& zm, PVZonesProcessor& zp_sel, PVZonesProcessor& zp_bg):
	QGraphicsScene(),
	_lines_view(backend, zm, zp_sel, zp_bg, this),
	_lib_view(*view_sp),
	_full_parallel_view(full_parallel_view),
	_selection_square(new PVSelectionSquare(this)),
	_zoom_y(1.0),
	_sm_p(sm_p),
	_zid_timer_render(PVZONEID_INVALID)
{
	_view_deleted = false;

	PVHive::get().register_actor(view_sp, _view_actor);

	setBackgroundBrush(QBrush(common::color_view_bg()));

	// this scrollbar is totally useless
	_full_parallel_view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

	connect(_full_parallel_view->horizontalScrollBar(), SIGNAL(sliderPressed()), this, SLOT(scrollbar_pressed_Slot()));
	connect(_full_parallel_view->horizontalScrollBar(), SIGNAL(sliderReleased()), this, SLOT(scrollbar_released_Slot()));

	PVParallelView::PVLinesView::list_zone_images_t images = _lines_view.get_zones_images();

	// Add ALL axes
	const PVZoneID nzones = _lines_view.get_number_of_managed_zones()+1;
	for (PVZoneID z = 0; z < nzones; z++) {
		add_axis(z);
	}

	_full_parallel_view->set_total_line_number(_lines_view.get_zones_manager().get_number_rows());

	_timer_render = new QTimer(this);
	_timer_render->setSingleShot(true);
	_timer_render->setInterval(RENDER_TIMER_TIMEOUT);
	connect(_timer_render, SIGNAL(timeout()), this, SLOT(render_all_zones_all_imgs()));

	/*
	_timer_render_single_zone = new QTimer(this);
	_timer_render_single_zone->setSingleShot(true);
	_timer_render_single_zone->setInterval(RENDER_TIMER_TIMEOUT);
	connect(_timer_render, SIGNAL(timeout()), this, SLOT(render_single_zone_all_imgs()));*/
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::~PVFullParallelScene
 *
 *****************************************************************************/
PVParallelView::PVFullParallelScene::~PVFullParallelScene()
{
	PVLOG_DEBUG("In PVFullParallelScene destructor\n");
	if (!_view_deleted) {
		common::get_lib_view(_lib_view)->remove_view(this);
	}
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::about_to_be_deleted
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::about_to_be_deleted()
{
	_view_deleted = true;
	graphics_view()->setDisabled(true);
	// Cancel everything!
	_lines_view.cancel_and_wait_all_rendering();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::add_axis
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::add_axis(PVZoneID const zone_id, int index)
{
	PVAxisGraphicsItem* axisw = new PVAxisGraphicsItem(_sm_p, lib_view(),
	                                                   _lib_view.get_axes_combination().get_axes_comb_id(zone_id));

	axisw->get_sliders_group()->set_axis_scale(_zoom_y);
	axisw->set_axis_length(_axis_length);

	connect(axisw->get_sliders_group(), SIGNAL(selection_sliders_moved(axis_id_t)),
	        this, SLOT(update_selection_from_sliders_Slot(axis_id_t)));
	connect(axisw, SIGNAL(new_zoomed_parallel_view(int)),
	        this, SLOT(emit_new_zoomed_parallel_view(int)));

	addItem(axisw);

	if (index < 0) {
		_axes.push_back(axisw);
	} else {
		_axes[index] = axisw;
	}
	//axisw->get_sliders_group()->add_selection_sliders(768, 1000);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::add_zone_image
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::add_zone_image()
{
	SingleZoneImagesItems single_zone_images_items;
	single_zone_images_items.sel = addPixmap(QPixmap());
	single_zone_images_items.bg = addPixmap(QPixmap());
	single_zone_images_items.bg->setOpacity(0.25);
	single_zone_images_items.bg->setZValue(0.0f);
	single_zone_images_items.sel->setZValue(1.0f);
	single_zone_images_items.img_tmp_sel = backend().create_image(PVParallelView::ZoneMaxWidth, PARALLELVIEW_ZT_BBITS);
	single_zone_images_items.img_tmp_bg  = backend().create_image(PVParallelView::ZoneMaxWidth, PARALLELVIEW_ZT_BBITS);
	_zones.push_back(single_zone_images_items);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::first_render
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::first_render()
{
	// AG & JBL: FIXME: This must be called after the view has been shown.
	// It seems like a magical QAbstractScrollbarArea stuff, investigation needed...
	PVParallelView::PVLinesView::list_zone_images_t images = _lines_view.get_zones_images();

	// Add visible zones
	_zones.reserve(images.size());
	for (PVZoneID zone_id = 0; zone_id < (PVZoneID) images.size() ; zone_id++) {
		add_zone_image();
	}

	update_zones_position(true, false);

	// Change view's internal counter
	update_selected_line_number();

	update_all();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::keyPressEvent
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::keyPressEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Space) {
		for (PVZoneID zone_id = _lines_view.get_first_visible_zone_index(); zone_id <= _lines_view.get_last_visible_zone_index(); zone_id++) {
			update_zone_pixmap_bgsel(zone_id);
		}
		event->accept();
	} else if (event->key() == Qt::Key_Home) {
		reset_zones_layout_to_default();
		update_all_with_timer();
		event->accept();
	}
	else if (event->key() == Qt::Key_Left) {
		if (event->modifiers() & Qt::ShiftModifier) {
			_selection_square->grow_horizontally();
		}
		else if (event->modifiers() & Qt::ControlModifier) {
			_selection_square->move_left_by_width();
		}
		else {
			_selection_square->move_left_by_step();
		}
		event->accept();
	}
	else if (event->key() == Qt::Key_Right) {
		if (event->modifiers() & Qt::ShiftModifier) {
			_selection_square->shrink_horizontally();
		}
		else if (event->modifiers() & Qt::ControlModifier) {
			_selection_square->move_right_by_width();
		}
		else {
			_selection_square->move_right_by_step();
		}
		event->accept();
	}
	else if (event->key() == Qt::Key_Up) {
		if (event->modifiers() & Qt::ShiftModifier) {
			_selection_square->grow_vertically();
		}
		else if (event->modifiers() & Qt::ControlModifier) {
			_selection_square->move_up_by_height();
		}
		else {
			_selection_square->move_up_by_step();
		}
		event->accept();
	}
	else if (event->key() == Qt::Key_Down) {
		if (event->modifiers() & Qt::ShiftModifier) {
			_selection_square->shrink_vertically();
		}
		else if (event->modifiers() & Qt::ControlModifier) {
			_selection_square->move_down_by_height();
		}
		else {
			_selection_square->move_down_by_step();
		}
		event->accept();
	}
	else {
		QGraphicsScene::keyPressEvent(event);
	}
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::mouseMoveEvent
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->buttons() == Qt::RightButton) {
		// Translate viewport
		QScrollBar *hBar = _full_parallel_view->horizontalScrollBar();
		hBar->setValue(hBar->value() + int(_translation_start_x - event->scenePos().x()));
		event->accept();
	}
	else if (!sliders_moving() && event->buttons() == Qt::LeftButton)
	{
		// trace square area
		_selection_square->end(event->scenePos().x(), event->scenePos().y());

		event->accept();
	}

	QGraphicsScene::mouseMoveEvent(event);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::mousePressEvent
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	QGraphicsScene::mousePressEvent(event);

	if (event->isAccepted()) {
		// a QGraphicsItem has already done something (usually a contextMenuEvent)
		return;
	}

	if (event->button() == Qt::RightButton) {
		// Store view position to compute translation
		_translation_start_x = event->scenePos().x();
		event->accept();
	} else if (event->button() == Qt::LeftButton) {


		/* setting the selection "square" to a "zero" square at mouse
		 * position and make it visible
		 */
		/*_selection_square_pos = event->scenePos();
		_selection_square->update_rect(QRectF(_selection_square_pos,
		                                      _selection_square_pos));
		_selection_square->show();*/
		_selection_square->begin(event->scenePos().x(), event->scenePos().y());
		event->accept();
	}
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::mouseReleaseEvent
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->button() == Qt::RightButton) {
		// translate zones
		translate_and_update_zones_position();
		event->accept();
	}
	else if (!sliders_moving()) {
		_selection_square->commit();
		event->accept();
	}

	QGraphicsScene::mouseReleaseEvent(event);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::pos_last_axis
 *
 *****************************************************************************/
int32_t PVParallelView::PVFullParallelScene::pos_last_axis() const
{
	const PVZoneID lastz = _lines_view.get_number_of_managed_zones()-1;
	int32_t pos = _lines_view.get_left_border_position_of_zone_in_scene(lastz);
	pos += _lines_view.get_zone_width(lastz);
	return pos;
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::process_selection
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::process_mouse_selection()
{
	unsigned int modifiers = (unsigned int) QApplication::keyboardModifiers();
	/* We don't care about a keypad button being pressed */
	modifiers &= ~Qt::KeypadModifier;

	/* Can't use a switch case here as Qt::ShiftModifier and Qt::ControlModifier aren't really
	 * constants */
	if (modifiers == (unsigned int) (Qt::ShiftModifier | Qt::ControlModifier)) {
		_view_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_INTERSECT_VOLATILE);
	}
	else
	if (modifiers == Qt::ControlModifier) {
		_view_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_SUBSTRACT_VOLATILE);
	}
	else
	if (modifiers == Qt::ShiftModifier) {
		_view_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_ADD_VOLATILE);
	}
	else {
		_view_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
	}

	/* Commit the previous volatile selection */
	_view_actor.call<FUNC(Picviz::PVView::commit_volatile_in_floating_selection)>();

	_view_actor.call<FUNC(Picviz::PVView::process_real_output_selection)>();
}

void PVParallelView::PVFullParallelScene::process_key_selection()
{
	_view_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);

	/* Commit the previous volatile selection */
	_view_actor.call<FUNC(Picviz::PVView::commit_volatile_in_floating_selection)>();

	_view_actor.call<FUNC(Picviz::PVView::process_real_output_selection)>();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::render_all_zones_all_imgs
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::render_all_zones_all_imgs()
{
	const uint32_t view_x = _full_parallel_view->horizontalScrollBar()->value();
	const uint32_t view_width = _full_parallel_view->width();
	_lines_view.render_all_zones_images(view_x, view_width, _zoom_y);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::render_single_zone_all_imgs
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::render_single_zone_all_imgs()
{
	assert(_zid_timer_render != PVZONEID_INVALID);
	if (!_lines_view.is_zone_drawn(_zid_timer_render)) {
		return;
	}

	_lines_view.render_single_zone_images(_zid_timer_render, _zoom_y);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::scale_zone_images
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::scale_zone_images(PVZoneID zone_id)
{
	const PVZoneID img_id = _lines_view.get_zone_index_offset(zone_id);
	PVParallelView::PVLinesView::list_zone_images_t& images = _lines_view.get_zones_images();

	PVBCIBackendImage& img_sel = *images[img_id].sel;
	PVBCIBackendImage& img_bg = *images[img_id].bg;

	const uint32_t zone_width = _lines_view.get_zone_width(zone_id);
	{
		PVBCIBackendImage& scaled_img = *_zones[img_id].img_tmp_bg;
		img_bg.resize_width(scaled_img, zone_width);
		_zones[img_id].bg->setPixmap(QPixmap::fromImage(scaled_img.qimage()));
	}
	{
		PVBCIBackendImage& scaled_img = *_zones[img_id].img_tmp_sel;
		img_sel.resize_width(scaled_img, zone_width);
		_zones[img_id].sel->setPixmap(QPixmap::fromImage(scaled_img.qimage()));
	}
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::scrollbar_pressed_Slot
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::scrollbar_pressed_Slot()
{
	_translation_start_x = (qreal) _full_parallel_view->horizontalScrollBar()->value();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::scrollbar_released_Slot
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::scrollbar_released_Slot()
{
	translate_and_update_zones_position();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::sliders_moving
 *
 *****************************************************************************/
bool PVParallelView::PVFullParallelScene::sliders_moving() const
{
	for (PVAxisGraphicsItem* axis : _axes) {
		if (axis->get_sliders_group()->sliders_moving()) {
			return true;
		}
	}
	return false;
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::translate_and_update_zones_position
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::translate_and_update_zones_position()
{
	assert(QThread::currentThread() == this->thread());
	uint32_t view_x = _full_parallel_view->horizontalScrollBar()->value();
	uint32_t view_width = _full_parallel_view->width();
	_lines_view.translate(view_x, view_width, _zoom_y);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_all
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_all()
{
	assert(QThread::currentThread() == this->thread());
	render_all_zones_all_imgs();
	update_selected_line_number();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_all_async
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_all_async()
{
	QMetaObject::invokeMethod(this, "update_all", Qt::QueuedConnection);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_all_with_timer
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_all_with_timer()
{
	_timer_render->start();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_selected_line_number
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_selected_line_number()
{
	const PVRow nlines = lib_view().get_real_output_selection().get_number_of_selected_lines_in_range(0, _lines_view.get_zones_manager().get_number_rows());
	graphics_view()->set_selected_line_number(nlines);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_new_selection
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_new_selection()
{
	assert(QThread::currentThread() == this->thread());
	// Change view's internal counter
	update_selected_line_number();

	const uint32_t view_x = _full_parallel_view->horizontalScrollBar()->value();
	const uint32_t view_width = _full_parallel_view->width();
	_lines_view.render_all_zones_sel_image(view_x, view_width, _zoom_y);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_new_selection_async
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_new_selection_async()
{
	QMetaObject::invokeMethod(this, "update_new_selection", Qt::QueuedConnection);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_number_of_zones
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_number_of_zones()
{
	assert(QThread::currentThread() == this->thread());
	const uint32_t view_x = _full_parallel_view->horizontalScrollBar()->value();
	const uint32_t view_width = _full_parallel_view->width();
	_lines_view.update_number_of_zones(view_x, view_width);
	PVZoneID const nb_zones = _lines_view.get_number_of_managed_zones();
	PVZoneID nb_zones_drawable = _lines_view.get_number_of_visible_zones();
	if ((PVZoneID) _zones.size() != nb_zones_drawable) {
		if ((PVZoneID) _zones.size() > nb_zones_drawable) {
			for (PVZoneID zone_id = nb_zones_drawable; zone_id < (PVZoneID) _zones.size(); zone_id++) {
				_zones[zone_id].remove(this);
			}
			_zones.resize(nb_zones_drawable);
		}
		else {
			_zones.reserve(nb_zones_drawable);
			for (PVZoneID zone_id = _zones.size(); zone_id < nb_zones_drawable; zone_id++) {
				add_zone_image();
			}
		}
	}

	axes_list_t new_axes;

	// there are nb_zones+1 axes
	new_axes.resize(nb_zones + 1, nullptr);

	const PVLinesView::list_zone_width_with_zoom_level_t &old_wz_list =
		_lines_view.get_list_of_zone_width_with_zoom_level();

	PVLinesView::list_zone_width_with_zoom_level_t new_wz_list;
	// use of a negative width to indicate an uninitialized entry
	new_wz_list.resize(nb_zones + 1, PVLinesView::ZoneWidthWithZoomLevel(-1, 0));

	/* to create the new axes list, already used axes are got back and
	 * moved to their new position in an array initialized to nullptr.
	 * Missing axes are deleted. Remainding nullptr entries in the new axes
	 * list are for new axes which are created.
	 *
	 * the same is done for PVLinesView::_list_zone_width_with_zoom_level
	 * to preserve kept zones widths.
	 */
	for (size_t i = 0; i < _axes.size(); ++i) {
		PVCol index = _lib_view.get_axes_combination().get_index_by_id(_axes[i]->get_axis_id());
		if (index == PVCOL_INVALID_VALUE) {
			// AG: this is really important to do this to force the
			// deletion of this PVAxisGraphicsItem object. Indeed,
			// removeItem will remove this object from the list of children
			// of the scene, and gives us the ownship of the object. Thus,
			// we are free to delete it afterwards.
			_axes[i]->get_sliders_group()->delete_own_selection_sliders();
			removeItem(_axes[i]);
			delete _axes[i];
		} else {
			new_axes[index] = _axes[i];
			new_axes[index]->update_axis_info();
			new_wz_list[index] = old_wz_list[i];
		}

		_axes[i] = nullptr;
	}

	_axes = new_axes;

	for (size_t i = 0; i < _axes.size(); ++i) {
		if (_axes[i] == nullptr) {
			add_axis(i, i);
		}

		if (new_wz_list[i].get_base_width() < 0) {
			// initialization of newly created zones widths
			new_wz_list[i] = PVLinesView::ZoneWidthWithZoomLevel();
		}
	}

	_lines_view.set_list_of_zone_width_with_zoom_level(new_wz_list);
	update_zones_position(true, false);

	set_enabled(true);
	update_all();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_number_of_zones_async
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_number_of_zones_async()
{
	QMetaObject::invokeMethod(this, "update_number_of_zones", Qt::QueuedConnection);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_scene
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_scene(bool recenter_view)
{
	QRectF old_scene_rect = sceneRect();
	QRectF items_bbox = itemsBoundingRect();

	QRectF new_scene_rect(_lines_view.get_left_border_of_scene() - 0.9*_full_parallel_view->width(), items_bbox.top(),
	                      _lines_view.get_right_border_of_scene() + 1.8*_full_parallel_view->width(), items_bbox.height() + SCENE_MARGIN);


	if (old_scene_rect.width() == new_scene_rect.width()) {
		/* QGraphicsView::centerOn(...) is not stable:
		 * centerOn(scene_center) may differ from scene_center. Thx Qt's guys!
		 */
		return;
	}

	QRect screen_rect = _full_parallel_view->viewport()->rect();
	qreal old_center_x = _full_parallel_view->mapToScene(screen_rect.center()).x();

	// set scene's bounding box because Qt never shrinks the sceneRect (see Qt Doc)
	setSceneRect(new_scene_rect);

	if (recenter_view) {
		// center's ordinate must always show axes names
		qreal new_center_y = items_bbox.top() + screen_rect.center().y();

		_full_parallel_view->centerOn(old_center_x, new_center_y);
	}
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_selection_from_sliders_Slot
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_selection_from_sliders_Slot(axis_id_t axis_id)
{
	PVZoneID zone_id = _lib_view.get_axes_combination().get_index_by_id(axis_id);
	_selection_square->clear();
	PVSelectionGenerator::compute_selection_from_sliders(
		_lines_view,
		zone_id,
	    _axes[zone_id]->get_selection_ranges(),
	    lib_view().get_volatile_selection()
	);

	process_mouse_selection();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_viewport
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_viewport()
{
	int screen_height = _full_parallel_view->viewport()->rect().height();

	QRectF axes_names_bbox_f;

	for(PVAxisGraphicsItem *axis : _axes) {
		axes_names_bbox_f |= axis->get_label_scene_bbox();
	}

	/* the bbox is extended to 0 to consider the offset between the labels and
	 * the top of the axis, and the offset 0 and the axis top
	 */
	axes_names_bbox_f.setBottom(0.);

	int labels_height = _full_parallel_view->mapFromScene(axes_names_bbox_f).boundingRect().height();
	_axis_length = PVCore::clamp(screen_height - (labels_height + SCENE_MARGIN),
	                             0, 1024);

	for(PVAxisGraphicsItem *axis : _axes) {
		axis->set_axis_length(_axis_length);
	}

	QRectF r = _selection_square->get_rect();

	if (r.isNull() == false) {
		/* if the selection rectangle exists, it must be unscaled (using
		 * the old y zoom factor)...
		 */
		r.setTop(r.top() / _zoom_y);
		r.setBottom(r.bottom() / _zoom_y);
	}

	_zoom_y = _axis_length / 1024.;

	// propagate this value to all PVSlidersGroup
	for(PVAxisGraphicsItem *axis : _axes) {
		axis->get_sliders_group()->set_axis_scale(_zoom_y);
	}

	if (r.isNull() == false) {
		// and it must be rescaled (using the new y zoom factor)
		r.setTop(r.top() * _zoom_y);
		r.setBottom(r.bottom() * _zoom_y);
		// AG: don't do an update_rect here since it will change the current selection!
		_selection_square->update_rect_no_commit(r);
	}
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_zone_pixmap_bg
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_zone_pixmap_bg(int zone_id)
{
	assert(_lines_view.is_zone_drawn(zone_id));

	PVParallelView::PVLinesView::list_zone_images_t& images = _lines_view.get_zones_images();
	const PVZoneID img_id = zone_id-_lines_view.get_first_visible_zone_index();

	// Check whether the image needs scaling.
	const uint32_t zone_width = _lines_view.get_zone_width(zone_id);

	PVBCIBackendImage& img_bg = *images[img_id].bg;

	if (img_bg.width() != zone_width) {
		return;
	}

	_zones[img_id].bg->setPixmap(QPixmap::fromImage(img_bg.qimage()));
	_zones[img_id].bg->setPos(QPointF(_lines_view.get_left_border_position_of_zone_in_scene(zone_id), 0));
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_zone_pixmap_bgsel
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_zone_pixmap_bgsel(int zone_id)
{
	update_zone_pixmap_bg(zone_id);
	update_zone_pixmap_sel(zone_id);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_zone_pixmap_sel
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_zone_pixmap_sel(int zone_id)
{
	const PVZoneID img_id = _lines_view.get_zone_index_offset(zone_id);
	PVParallelView::PVLinesView::list_zone_images_t& images = _lines_view.get_zones_images();

	// Check whether the image needs scaling.
	const uint32_t zone_width = _lines_view.get_zone_width(zone_id);

	PVBCIBackendImage& img_sel = *images[img_id].sel;

	if (img_sel.width() != zone_width) {
		return;
	}

	_zones[img_id].sel->setPixmap(QPixmap::fromImage(img_sel.qimage()));
	_zones[img_id].sel->setPos(QPointF(_lines_view.get_left_border_position_of_zone_in_scene(zone_id), 0));
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_zones_position
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_zones_position(bool update_all, bool scale)
{
	if (scale) {
		//BENCH_START(update);
		for (PVZoneID zone_id = _lines_view.get_first_visible_zone_index(); zone_id <= _lines_view.get_last_visible_zone_index(); zone_id++) {
			scale_zone_images(zone_id);
		}
		//BENCH_END(update, "update_zone_pixmap", 1, 1, 1, 1);
	}

	// We start by updating all axes positions
	PVZoneID nzones = (PVZoneID) _lines_view.get_number_of_managed_zones()+1;
	uint32_t pos = 0;

	_axes[0]->setPos(QPointF(0, 0));
	PVZoneID z = 1;
	if (!update_all) {
		uint32_t view_x = _full_parallel_view->horizontalScrollBar()->value();
		z = _lines_view.get_zone_from_scene_pos(view_x) + 1;
	}
	for (; z < nzones; z++) {
		if (z < nzones-1) {
			pos = _lines_view.get_left_border_position_of_zone_in_scene(z);
		}
		else {
			// Special case for last axis
			pos += _lines_view.get_zone_width(z-1);
		};

		_axes[z]->setPos(QPointF(pos - PVParallelView::AxisWidth, 0));
	}

	// We now update all zones positions
	for (PVZoneID z = _lines_view.get_first_visible_zone_index(); z <= _lines_view.get_last_visible_zone_index(); z++) {
		_zones[_lines_view.get_zone_index_offset(z)].setPos(QPointF(_lines_view.get_left_border_position_of_zone_in_scene(z), 0));
	}
	
	// It's time to refresh the current selection_square
	_selection_square->update_position();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::wheelEvent
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::wheelEvent(QGraphicsSceneWheelEvent* event)
{
	const int delta = event->delta();
	const int old_view_x = graphics_view()->horizontalScrollBar()->value();

	// Get the zone_id of the zone under mouse cursor
	const QPointF mouse_scene_pt = event->scenePos();
	const PVZoneID zmouse = _lines_view.get_zone_from_scene_pos(mouse_scene_pt.x());

	int32_t const mouse_scene_x = mouse_scene_pt.x();
	int32_t const zmouse_x = _lines_view.get_left_border_position_of_zone_in_scene(zmouse);

	const double rel_pos = (double)(mouse_scene_x-zmouse_x)/((double)_lines_view.get_zone_width(zmouse));

	if (event->modifiers() & Qt::ShiftModifier) {
		/* we do not want the QGraphicsScene's behaviour using shift (+ other modifier) + wheel:
		 * a vertical scroll.
		 */
		event->accept();
	} else if (event->modifiers() == Qt::ControlModifier) {
		// Local zoom when the 'Ctrl' key is pressed

		if (delta < 0) {
			_lines_view.decrease_base_zoom_level_of_zone(zmouse);
		}
		else {
			_lines_view.increase_base_zoom_level_of_zone(zmouse);
		}
		
		update_viewport();
		update_zones_position(true, true);

		// Compute new view_x
		const int32_t zmouse_new_x = _lines_view.get_left_border_position_of_zone_in_scene(zmouse);
		int32_t const new_mouse_scene_x = (int32_t) ((double)zmouse_new_x + rel_pos*(double)_lines_view.get_zone_width(zmouse));
		int32_t const new_view_x = old_view_x + (new_mouse_scene_x - mouse_scene_x);

		graphics_view()->horizontalScrollBar()->setValue(new_view_x);

		update_scene(false);

		/*_timer_render_one_zone->stop();
		_zid_timer_render = zone_id;
		_timer_render_one_zone->start();*/
		_lines_view.render_single_zone_images(zmouse, _zoom_y);

		event->accept();
	}
	else if (event->modifiers() == Qt::NoModifier) {
		// Get mouse position in the scene
		const QPointF mouse_scene_pt = event->scenePos();

		// Get the relative position to the closest left axis
		PVZoneID const zmouse = _lines_view.get_zone_from_scene_pos(mouse_scene_pt.x());
		int32_t const mouse_scene_x = mouse_scene_pt.x();
		int32_t const zmouse_x = _lines_view.get_left_border_position_of_zone_in_scene(zmouse);
		double rel_pos = (double)(mouse_scene_x-zmouse_x)/((double)_lines_view.get_zone_width(zmouse));

 		//Global zoom
		if (delta < 0) {
			_lines_view.decrease_global_zoom_level();
		}
		else if (delta >0) {
			_lines_view.increase_global_zoom_level();
		}
		
		update_viewport();
		update_zones_position(true, true);

		// Compute new view_x
		const int32_t zmouse_new_x = _lines_view.get_left_border_position_of_zone_in_scene(zmouse);
		int32_t const new_mouse_scene_x = (int32_t) ((double)zmouse_new_x + rel_pos*(double)_lines_view.get_zone_width(zmouse));
		int32_t const new_view_x = old_view_x + (new_mouse_scene_x - mouse_scene_x);

		graphics_view()->horizontalScrollBar()->setValue(new_view_x);

		update_scene(false);
		_timer_render->start();
		event->accept();
	}
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::zr_bg_finished
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::zr_bg_finished(PVZoneRenderingBase_p zr, int zid)
{
	assert(QThread::currentThread() == this->thread());
	if (_view_deleted) {
		return;
	}

	if (!_lines_view.is_zone_drawn(zid)) {
		// This can occur if some events have been posted by a previous translation that is no longer valid!
		return;
	}

	if (zr) {
		const PVZoneID img_id = _lines_view.get_zone_index_offset(zid);
		PVParallelView::PVLinesView::list_zone_images_t& images = _lines_view.get_zones_images();
		if (zr == images[img_id].last_zr_bg) {
			images[img_id].last_zr_bg.reset();
		}

		bool should_cancel = zr->should_cancel();
		if (should_cancel) {
			// Cancellation may have occured between the event posted in Qt's main loop and this call!
			return;
		}
	}

	update_zone_pixmap_bg(zid);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::zr_sel_finished
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::zr_sel_finished(PVZoneRenderingBase_p zr, int zid)
{
	assert(QThread::currentThread() == this->thread());
	if (_view_deleted) {
		return;
	}

	if (!_lines_view.is_zone_drawn(zid)) {
		// This can occur if some events have been posted by a previous translation that is no longer valid!
		return;
	}

	if (zr) {
		const PVZoneID img_id = _lines_view.get_zone_index_offset(zid);
		PVParallelView::PVLinesView::list_zone_images_t& images = _lines_view.get_zones_images();
		if (zr == images[img_id].last_zr_sel) {
			images[img_id].last_zr_sel.reset();
		}

		bool should_cancel = zr->should_cancel();
		if (should_cancel) {
			// Cancellation may have occured between the event posted in Qt's main loop and this call!
			return;
		}
	}

	update_zone_pixmap_sel(zid);
}

/******************************************************************************
 * PVParallelView::PVFullParallelScene::reset_zones_layout_to_default
 *****************************************************************************/

void PVParallelView::PVFullParallelScene::reset_zones_layout_to_default()
{
	QRect screen_rect = _full_parallel_view->viewport()->rect();

	bool fit_in = _lines_view.initialize_zones_width(screen_rect.width());

	update_viewport();
	update_zones_position(true, true);

	update_scene(false);

	// time to replace the viewport at the right position
	QRectF items_bbox = itemsBoundingRect();

	qreal view_center_x;
	if (fit_in) {
		// center zones
		view_center_x = sceneRect().center().x();
	} else {
		// align zone to the left
		view_center_x = (screen_rect.width() / 2) - SCENE_MARGIN;
	}

	_full_parallel_view->centerOn(view_center_x,
	                              items_bbox.top() + screen_rect.center().y());
}

