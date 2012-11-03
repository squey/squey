/**
 * \file PVParallelScene.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <picviz/PVStateMachine.h>
#include <picviz/PVView.h>

#include <picviz/widgets/editors/PVAxisIndexEditor.h>

#include <pvhive/PVCallHelper.h>

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

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::PVFullParallelScene
 *
 *****************************************************************************/
PVParallelView::PVFullParallelScene::PVFullParallelScene(PVFullParallelView* parallel_view, Picviz::PVView_sp& view_sp, PVParallelView::PVSlidersManager_p sm_p, PVBCIDrawingBackend& backend, PVZonesManager const& zm, PVZonesProcessor& zp_sel, PVZonesProcessor& zp_bg):
	QGraphicsScene(),
	_lines_view(backend, zm, zp_sel, zp_bg, this),
	_lib_view(*view_sp),
	_parallel_view(parallel_view),
	_selection_square(new PVParallelView::PVSelectionSquareGraphicsItem(this)),
	_selection_generator(_lines_view),
	_zoom_y(1.0),
	_sm_p(sm_p)
{
	PVHive::get().register_actor(view_sp, _view_actor);

	setBackgroundBrush(QBrush(common::color_view_bg()));

	// this scrollbar is totally useless
	_parallel_view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

	connect(_parallel_view->horizontalScrollBar(), SIGNAL(sliderPressed()), this, SLOT(scrollbar_pressed_Slot()));
	connect(_parallel_view->horizontalScrollBar(), SIGNAL(sliderReleased()), this, SLOT(scrollbar_released_Slot()));
	connect(_selection_square, SIGNAL(commit_volatile_selection()), this, SLOT(commit_volatile_selection_Slot()));

	PVParallelView::PVLinesView::list_zone_images_t images = _lines_view.get_zones_images();

	// Add ALL axes
	const PVZoneID nzones = _lines_view.get_number_of_zones()+1;
	for (PVZoneID z = 0; z < nzones; z++) {
		add_axis(z);
	}

	_parallel_view->set_total_line_number(_lines_view.get_zones_manager().get_number_rows());

	_timer_render = new QTimer(this);
	_timer_render->setSingleShot(true);
	_timer_render->setInterval(100);
	connect(_timer_render, SIGNAL(timeout()), this, SLOT(render_all_zones_all_imgs()));
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::~PVFullParallelScene
 *
 *****************************************************************************/
PVParallelView::PVFullParallelScene::~PVFullParallelScene()
{
	PVLOG_INFO("In PVFullParallelScene destructor\n");
	//common::get_lib_view(_lib_view)->remove_view(this);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::about_to_be_deleted
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::about_to_be_deleted()
{
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
	ZoneImages zi;
	zi.sel = addPixmap(QPixmap());
	zi.bg = addPixmap(QPixmap());
	zi.bg->setOpacity(0.25);
	zi.bg->setZValue(0.0f);
	zi.sel->setZValue(1.0f);
	zi.img_tmp_sel = backend().create_image(PVParallelView::ZoneMaxWidth, PARALLELVIEW_ZT_BBITS);
	zi.img_tmp_bg  = backend().create_image(PVParallelView::ZoneMaxWidth, PARALLELVIEW_ZT_BBITS);
	_zones.push_back(zi);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::commit_volatile_selection_Slot
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::commit_volatile_selection_Slot()
{
	_selection_square->finished();
	QRectF srect = _selection_square->rect();
	PVLOG_INFO("srect start: %f, srect end: %f\n", srect.x(), srect.x() + srect.width());
	// Too much on the left dude!
	if (srect.x() + srect.width() <= 0) {
		return;
	}

	// Too much on the right, stop drinking!
	const int32_t pos_end = pos_last_axis();
	if (srect.x() >= pos_end) {
		return;
	}

	const PVZoneID zone_id_start = _lines_view.get_zone_from_scene_pos(srect.x());
	const PVZoneID zone_id_end = _lines_view.get_zone_from_scene_pos(srect.x() + srect.width());

	lib_view().get_volatile_selection().select_none();
	for (PVZoneID z = zone_id_start; z <= zone_id_end; z++) {
		QRect r = map_to_axis(z, srect);
		r.setX(picviz_max(0, r.x()));
		r.setRight(picviz_min(pos_end-1, r.right()));
		_selection_generator.compute_selection_from_rect(z, r, lib_view().get_volatile_selection());
	}

	store_selection_square();

	process_selection();
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
	for (PVZoneID z = 0; z < (PVZoneID) images.size() ; z++) {
		add_zone_image();
	}

	update_zones_position(true, false);

	// Change view's internal counter
	const PVRow nlines = lib_view().get_real_output_selection().get_number_of_selected_lines_in_range(0, _lines_view.get_zones_manager().get_number_rows());
	graphics_view()->set_selected_line_number(nlines);

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
		for (PVZoneID zone_id = _lines_view.get_first_drawn_zone(); zone_id <= _lines_view.get_last_drawn_zone(); zone_id++) {
			update_zone_pixmap_bgsel(zone_id);
		}
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
		QScrollBar *hBar = _parallel_view->horizontalScrollBar();
		hBar->setValue(hBar->value() + int(_translation_start_x - event->scenePos().x()));
		event->accept();
	}
	else if (!sliders_moving() && event->buttons() == Qt::LeftButton)
	{
		// trace square area
		QPointF top_left(qMin(_selection_square_pos.x(), event->scenePos().x()), qMin(_selection_square_pos.y(), event->scenePos().y()));
		QPointF bottom_right(qMax(_selection_square_pos.x(), event->scenePos().x()), qMax(_selection_square_pos.y(), event->scenePos().y()));

		_selection_square->update_rect(QRectF(top_left, bottom_right));
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
		_selection_square_pos = event->scenePos();
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
		if (_selection_square_pos == event->scenePos()) {
			// Remove selection
			_selection_square->clear_rect();
		}
		_selection_square->finished();
		commit_volatile_selection_Slot();
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
	const PVZoneID lastz = _lines_view.get_number_of_zones()-1;
	int32_t pos = _lines_view.get_zone_absolute_pos(lastz);
	pos += _lines_view.get_zone_width(lastz);
	return pos;
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::process_selection
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::process_selection()
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

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::render_all_zones_all_imgs
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::render_all_zones_all_imgs()
{
	PVLOG_INFO("!!!!!!!!!!in render_all_zones_all_imgs!!!!!!\n");
	const uint32_t view_x = _parallel_view->horizontalScrollBar()->value();
	const uint32_t view_width = _parallel_view->width();
	_lines_view.render_all_zones_all_imgs(view_x, view_width, _zoom_y);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::scale_zone_images
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::scale_zone_images(PVZoneID zone_id)
{
	const PVZoneID img_id = _lines_view.get_zone_image_idx(zone_id);
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
	_translation_start_x = (qreal) _parallel_view->horizontalScrollBar()->value();
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
 * PVParallelView::PVFullParallelScene::store_selection_square
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::store_selection_square()
{
	PVZoneID& zone_id1 = _selection_barycenter.zone_id1;
	PVZoneID& zone_id2 = _selection_barycenter.zone_id2;
	double& factor1 = _selection_barycenter.factor1;
	double& factor2 = _selection_barycenter.factor2;

	uint32_t abs_left = _selection_square->rect().topLeft().x();
	uint32_t abs_right = _selection_square->rect().bottomRight().x();

	zone_id1 = _lines_view.get_zone_from_scene_pos(abs_left);
	uint32_t z1_width = _lines_view.get_zone_width(zone_id1);
	uint32_t alpha = map_to_axis(zone_id1, QPointF(abs_left, 0)).x();
	factor1 = (double) alpha / z1_width;

	zone_id2 = _lines_view.get_zone_from_scene_pos(abs_right);
	uint32_t z2_width = _lines_view.get_zone_width(zone_id2);
	uint32_t beta = map_to_axis(zone_id2, QPointF(abs_right, 0)).x();
	factor2 = (double) beta / z2_width;
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::translate_and_update_zones_position
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::translate_and_update_zones_position()
{
	uint32_t view_x = _parallel_view->horizontalScrollBar()->value();
	uint32_t view_width = _parallel_view->width();
	_lines_view.translate(view_x, view_width, _zoom_y);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_all
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_all()
{
	render_all_zones_all_imgs();
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
 * PVParallelView::PVFullParallelScene::update_new_selection
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_new_selection()
{
	// Change view's internal counter
	const PVRow nlines = lib_view().get_real_output_selection().get_number_of_selected_lines_in_range(0, _lines_view.get_zones_manager().get_number_rows());
	graphics_view()->set_selected_line_number(nlines);

	const uint32_t view_x = _parallel_view->horizontalScrollBar()->value();
	const uint32_t view_width = _parallel_view->width();
	_lines_view.render_all_imgs_sel(view_x, view_width, _zoom_y);
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
	const uint32_t view_x = _parallel_view->horizontalScrollBar()->value();
	const uint32_t view_width = _parallel_view->width();
	_lines_view.update_number_of_zones(view_x, view_width);
	PVZoneID const nb_zones = _lines_view.get_zones_manager().get_number_of_zones();
	PVZoneID nb_zones_drawable = _lines_view.get_nb_drawable_zones();
	if ((PVZoneID) _zones.size() != nb_zones_drawable) {
		if ((PVZoneID) _zones.size() > nb_zones_drawable) {
			for (PVZoneID z = nb_zones_drawable; z < (PVZoneID) _zones.size(); z++) {
				_zones[z].remove(this);
			}
			_zones.resize(nb_zones_drawable);
		}
		else {
			_zones.reserve(nb_zones_drawable);
			for (PVZoneID z = _zones.size(); z < nb_zones_drawable; z++) {
				add_zone_image();
			}
		}
	}

	axes_list_t new_axes;

	// there are nb_zones+1 axes
	new_axes.resize(nb_zones + 1, nullptr);

	/* to create the new axes list, already used axes are got back and
	 * moved to their new position in an array initialized to nullptr.
	 * Missing axes are deleted. Remainding nullptr entries in the new axes
	 * list are for new axes which are created.
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
		}

		_axes[i] = nullptr;
	}

	_axes = new_axes;

	for (size_t i = 0; i < _axes.size(); ++i) {
		if (_axes[i] == nullptr) {
			add_axis(i, i);
		}
	}

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
void PVParallelView::PVFullParallelScene::update_scene(QGraphicsSceneWheelEvent* event)
{
	QRectF old_scene_rect = sceneRect();
	QRectF items_bbox = itemsBoundingRect();
	QRectF new_scene_rect(items_bbox.left() - SCENE_MARGIN, items_bbox.top(),
	                      items_bbox.right() + (2*SCENE_MARGIN), items_bbox.bottom() + SCENE_MARGIN);

	if (old_scene_rect.width() == new_scene_rect.width()) {
		/* QGraphicsView::centerOn(...) is not stable:
		 * centerOn(scene_center) may differ from scene_center. Thx Qt's guys!
		 */
		return;
	}

	QRect screen_rect = _parallel_view->viewport()->rect();
	QPointF old_center = _parallel_view->mapToScene(screen_rect.center());

	// set scene's bounding box because Qt never shrinks the sceneRect (see Qt Doc)
	setSceneRect(new_scene_rect);

	qreal new_center_x;

	if (event == nullptr) {
		// due to a resize event
		new_center_x = old_center.x();
	} else {
		qreal mouse_x = event->scenePos().x();
		qreal dx = old_center.x() - mouse_x;
		qreal rel_mouse_x = mouse_x / (qreal)old_scene_rect.width();
		qreal new_mouse_x = rel_mouse_x * (qreal)new_scene_rect.width();

		new_center_x = new_mouse_x + dx;
	}

	// center's ordinate must always show axes names
	qreal new_center_y = items_bbox.top() + screen_rect.center().y();

	_parallel_view->centerOn(new_center_x, new_center_y);

}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_selection_from_sliders_Slot
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_selection_from_sliders_Slot(axis_id_t axis_id)
{
	PVZoneID zone_id = _lib_view.get_axes_combination().get_index_by_id(axis_id);
	_selection_square->clear_rect();
	uint32_t nb_select = _selection_generator.compute_selection_from_sliders(zone_id, _axes[zone_id]->get_selection_ranges(), lib_view().get_volatile_selection());
	_parallel_view->set_selected_line_number(nb_select);

	process_selection();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_selection_square
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_selection_square()
{
	PVZoneID zone_id1 = _selection_barycenter.zone_id1;
	PVZoneID zone_id2 = _selection_barycenter.zone_id2;
	if ((zone_id1 == PVZONEID_INVALID) || (zone_id2 == PVZONEID_INVALID)) {
		return;
	}

	if (zone_id1 >= _lines_view.get_zones_manager().get_number_of_zones() ||
	    zone_id2 >= _lines_view.get_zones_manager().get_number_of_zones()) {
		clear_selection_square();
		return;
	}

	double factor1 = _selection_barycenter.factor1;
	double factor2 = _selection_barycenter.factor2;

	uint32_t new_left = _lines_view.get_zone_absolute_pos(zone_id1) + (double) _lines_view.get_zone_width(zone_id1) * factor1;
	uint32_t new_right = _lines_view.get_zone_absolute_pos(zone_id2) + (double) _lines_view.get_zone_width(zone_id2) * factor2;
	uint32_t abs_top = _selection_square->rect().topLeft().y();
	uint32_t abs_bottom = _selection_square->rect().bottomRight().y();

	_selection_square->setRect(QRectF(QPointF(new_left, abs_top), QPointF(new_right, abs_bottom)));
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::update_viewport
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::update_viewport()
{
	int screen_height = _parallel_view->viewport()->rect().height();

	QRectF axes_names_bbox_f;

	for(PVAxisGraphicsItem *axis : _axes) {
		axes_names_bbox_f |= axis->get_label_scene_bbox();
	}

	/* the bbox is extended to 0 to consider the offset between the labels and
	 * the top of the axis, and the offset 0 and the axis top
	 */
	axes_names_bbox_f.setBottom(0.);

	int labels_height = _parallel_view->mapFromScene(axes_names_bbox_f).boundingRect().height();
	_axis_length = PVCore::clamp(screen_height - (labels_height + SCENE_MARGIN),
	                             0, 1024);

	for(PVAxisGraphicsItem *axis : _axes) {
		axis->set_axis_length(_axis_length);
	}

	QRectF r = _selection_square->rect();

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
		//_selection_square->update_rect(r);
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
	const PVZoneID img_id = zone_id-_lines_view.get_first_drawn_zone();

	// Check whether the image needs scaling.
	const uint32_t zone_width = _lines_view.get_zone_width(zone_id);

	PVBCIBackendImage& img_bg = *images[img_id].bg;

	if (img_bg.width() != zone_width) {
		return;
	}

	_zones[img_id].bg->setPixmap(QPixmap::fromImage(img_bg.qimage()));
	_zones[img_id].bg->setPos(QPointF(_lines_view.get_zone_absolute_pos(zone_id), 0));
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
	const PVZoneID img_id = _lines_view.get_zone_image_idx(zone_id);
	PVParallelView::PVLinesView::list_zone_images_t& images = _lines_view.get_zones_images();

	// Check whether the image needs scaling.
	const uint32_t zone_width = _lines_view.get_zone_width(zone_id);

	PVBCIBackendImage& img_sel = *images[img_id].sel;

	if (img_sel.width() != zone_width) {
		return;
	}

	_zones[img_id].sel->setPixmap(QPixmap::fromImage(img_sel.qimage()));
	_zones[img_id].sel->setPos(QPointF(_lines_view.get_zone_absolute_pos(zone_id), 0));
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
		for (PVZoneID zone_id = _lines_view.get_first_drawn_zone(); zone_id <= _lines_view.get_last_drawn_zone(); zone_id++) {
			scale_zone_images(zone_id);
		}
		//BENCH_END(update, "update_zone_pixmap", 1, 1, 1, 1);
	}

	// Update axes position
	PVZoneID nzones = (PVZoneID) _lines_view.get_zones_manager().get_number_of_zones()+1;
	uint32_t pos = 0;

	_axes[0]->setPos(QPointF(0, 0));
	PVZoneID z = 1;
	if (!update_all) {
		uint32_t view_x = _parallel_view->horizontalScrollBar()->value();
		z = _lines_view.get_zone_from_scene_pos(view_x) + 1;
	}
	for (; z < nzones; z++) {
		if (z < nzones-1) {
			pos = _lines_view.get_zone_absolute_pos(z);
		}
		else {
			// Special case for last axis
			pos += _lines_view.get_zone_width(z-1);
		};

		_axes[z]->setPos(QPointF(pos - PVParallelView::AxisWidth, 0));
	}

	for (PVZoneID z = _lines_view.get_first_drawn_zone(); z <= _lines_view.get_last_drawn_zone(); z++) {
		_zones[_lines_view.get_zone_image_idx(z)].setPos(QPointF(_lines_view.get_zone_absolute_pos(z), 0));
	}
		
	update_selection_square();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::wheelEvent
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::wheelEvent(QGraphicsSceneWheelEvent* event)
{
	int zoom = event->delta() / 2;
	
	// In case of a negative zoom
	if(zoom < 0) {
		zoom = picviz_max(zoom, -PVParallelView::ZoneMinWidth);
	}

	const QPointF mouse_scene_pt = event->scenePos();
	
	// We get the Zone_Id under the current mouse position
	PVZoneID mouse_zone_id = _lines_view.get_zone_from_scene_pos(mouse_scene_pt.x());
	
	// We test if it a LOCAL zoom (applies only to a zone) or a GLOBAL zoom
	if (event->modifiers() == Qt::ControlModifier) {
		// Local zoom
		const PVZoneID zone_id = mouse_zone_id;

		uint32_t z_width = _lines_view.get_zone_width(zone_id);
		if (_lines_view.set_zone_width(zone_id, z_width+zoom)) {
			update_viewport();
			update_zones_position(true, true);
			update_scene(event);

			_lines_view.render_zone_all_imgs(zone_id, _zoom_y);
		}
		event->accept();
	}
	else if (event->modifiers() == Qt::NoModifier) {
		//Global zoom
		if (_lines_view.set_all_zones_width([=](uint32_t width) { return width+zoom; })) {
			// at least one zone's width has been changed
			update_viewport();
			update_zones_position(true, true);
			update_scene(event);
		}
		_timer_render->start();
		event->accept();
	}
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::zr_bg_finished
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::zr_bg_finished(void* zr, int zone_id)
{
	update_zone_pixmap_bg(zone_id);

	if (zr) {
		PVRenderingPipeline::free_zr(reinterpret_cast<PVZoneRenderingBase*>(zr));

		const PVZoneID img_id = _lines_view.get_zone_image_idx(zone_id);
		PVParallelView::PVLinesView::list_zone_images_t& images = _lines_view.get_zones_images();
		images[img_id].last_zr_bg = nullptr;
	}
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelScene::zr_sel_finished
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelScene::zr_sel_finished(void* zr, int zone_id)
{
	update_zone_pixmap_sel(zone_id);

	if (zr) {
		PVRenderingPipeline::free_zr(reinterpret_cast<PVZoneRenderingBase*>(zr));
		
		const PVZoneID img_id = _lines_view.get_zone_image_idx(zone_id);
		PVParallelView::PVLinesView::list_zone_images_t& images = _lines_view.get_zones_images();
		images[img_id].last_zr_sel = nullptr;
	}
}


