
#include <pvparallelview/PVSelectionAxisSliders.h>
#include <pvparallelview/PVSlidersGroup.h>

#include <QGraphicsScene>

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::PVSelectionAxisSliders
 *****************************************************************************/

PVParallelView::PVSelectionAxisSliders::PVSelectionAxisSliders(QGraphicsItem *parent,
                                                               PVSlidersGroup *group) :
	PVAbstractAxisSliders(parent, group),
	_ssd_obs(this),
	_ssu_obs(this)
{
	setHandlesChildEvents(false);
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::initialize
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::initialize(PVParallelView::PVSlidersManager_p sm_p,
                                                        id_t id,
                                                        uint32_t y_min, uint32_t y_max)
{
	_sliders_manager_p = sm_p;
	_id = id;

	_sl_min = new PVAxisSlider(0, 1024, y_min);
	_sl_max = new PVAxisSlider(0, 1024, y_max);

	addToGroup(_sl_min);
	addToGroup(_sl_max);

	// set positions in their parent, not in the QGraphicsScene
	_sl_min->setPos(0, y_min);
	_sl_max->setPos(0, y_max);

	connect(_sl_min, SIGNAL(slider_moved()), this, SLOT(do_sliders_moved()));
	connect(_sl_max, SIGNAL(slider_moved()), this, SLOT(do_sliders_moved()));

	_text = new QGraphicsSimpleTextItem("    Range Selection");
	addToGroup(_text);
	_text->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
	_text->setBrush(Qt::white);
	_text->hide();

	PVHive::PVHive::get().register_func_observer(sm_p,
	                                             _ssd_obs);

	PVHive::PVHive::get().register_func_observer(sm_p,
	                                             _ssu_obs);
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::paint
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::paint(QPainter *painter,
                                              const QStyleOptionGraphicsItem */*option*/,
                                              QWidget */*widget*/)
{
	if (is_moving()) {
		painter->save();

		painter->setCompositionMode(QPainter::RasterOp_SourceXorDestination);

		QPen new_pen(Qt::white);
		new_pen.setWidth(0);
		painter->setPen(new_pen);
		painter->drawLine(0, _sl_min->value(), 0, _sl_max->value());

		_text->setPos(0, _sl_min->value());
		_text->show();

		painter->restore();
	} else {
		_text->hide();
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::do_sliders_moved
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::do_sliders_moved()
{
	emit sliders_moved();
	PVHive::call<FUNC(PVSlidersManager::update_selection_sliders)>(_sliders_manager_p,
	                                                               _group->get_axe_id(),
	                                                               _id,
	                                                               _sl_min->value(),
	                                                               _sl_max->value());
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::selection_sliders_del_obs::update
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::selection_sliders_del_obs::update(arguments_deep_copy_type const& args) const
{
	const axe_id_t &axe_id = std::get<0>(args);
	PVSlidersManager::id_t id = std::get<1>(args);

	if ((axe_id == _parent->_group->get_axe_id()) && (id == _parent->_id)) {
		_parent->group()->removeFromGroup(_parent);
		_parent->scene()->removeItem(_parent);
	}
}

/*****************************************************************************
 * PVParallelView::PVSelectionAxisSliders::selection_sliders_update_obs::update
 *****************************************************************************/

void PVParallelView::PVSelectionAxisSliders::selection_sliders_update_obs::update(arguments_deep_copy_type const& args) const
{
	const axe_id_t &axe_id = std::get<0>(args);
	PVSlidersManager::id_t id = std::get<1>(args);

	if ((axe_id == _parent->_group->get_axe_id()) && (id == _parent->_id)) {
		_parent->_sl_min->set_value(std::get<2>(args));
		_parent->_sl_max->set_value(std::get<3>(args));

		emit _parent->sliders_moved();
	}
}
