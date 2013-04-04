/**
 * \file PVAxisGraphicsItem.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <pvkernel/widgets/PVUtils.h>

#include <picviz/PVAxis.h>
#include <picviz/PVView.h>
#include <picviz/PVMapping.h>

#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVAxisLabel.h>

#include <QApplication>
#include <QDesktopWidget>
#include <QPainter>
#include <QGraphicsScene>

static void make_min_max_text(const QFont &font, const QString &text,
                              QString &elided_text, QString &tooltip_text,
                              int elided_width, int tooltip_width)
{
	elided_text = QFontMetrics(font).elidedText(text, Qt::ElideRight, elided_width);
	tooltip_text = text;
	PVWidgets::PVUtils::html_word_wrap_text(tooltip_text, font, tooltip_width);
}

namespace PVParallelView
{

namespace __impl
{

class PVMinMaxHelpEventFilter : public QObject
{
public:
	PVMinMaxHelpEventFilter(PVAxisGraphicsItem* parent):
	QObject(parent)
	{}

protected:
	bool eventFilter(QObject *obj, QEvent *ev)
	{
		QGraphicsTextItem* gti = static_cast<QGraphicsTextItem*>(obj);
		switch (ev->type()) {
		case QEvent::ToolTip:
			// agi_parent()->label_button_pressed(gti, static_cast<QHelpEvent*>(ev));
			std::cout << "######### POUET" << std::endl;
			break;
		default:
			std::cout << "@@@@@@@@@ " << ev->type() << std::endl;
			break;
		}

		return false;
	};

private:
	inline PVAxisGraphicsItem* agi_parent()
	{
		assert(qobject_cast<PVAxisGraphicsItem*>(parent()));
		return static_cast<PVAxisGraphicsItem*>(parent());
	}
};

}

}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem
 *****************************************************************************/

PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem(PVParallelView::PVSlidersManager_p sm_p,
                                                       Picviz::PVView const& view, const axis_id_t &axis_id) :
	_sliders_manager_p(sm_p),
	_axis_id(axis_id),
	_lib_view(view)
{
	_event_filter = new __impl::PVMinMaxHelpEventFilter(this);
	installEventFilter(_event_filter);

	// This is needed to let the children of the group handle their events.
	setHandlesChildEvents(false);

	// the sliders must be over all other QGraphicsItems
	setZValue(1.e42);

	_sliders_group = new PVSlidersGroup(sm_p, axis_id, this);

	addToGroup(get_sliders_group());
	get_sliders_group()->setPos(PARALLELVIEW_AXIS_WIDTH / 2, 0.);

	_label = new PVAxisLabel(view, _sliders_group);
	addToGroup(_label);
	_label->rotate(-45.);
	_label->setPos(0, - 6 * axis_extend);

	_axis_min_value = new QGraphicsTextItem(this);
	_axis_min_value->installEventFilter(_event_filter);
	addToGroup(_axis_min_value);

	_axis_max_value = new QGraphicsTextItem(this);
	_axis_max_value->installEventFilter(_event_filter);
	addToGroup(_axis_max_value);

	_layer_min_value = new QGraphicsTextItem(this);
	QFont font = _layer_min_value->font();
	font.setStyle(QFont::StyleItalic);
	_layer_min_value->setFont(font);
	_layer_min_value->installEventFilter(_event_filter);
	addToGroup(_layer_min_value);

	_layer_max_value = new QGraphicsTextItem(this);
	font = _layer_max_value->font();
	font.setStyle(QFont::StyleItalic);
	_layer_max_value->setFont(font);
	_layer_max_value->installEventFilter(_event_filter);
	addToGroup(_layer_max_value);

	connect(_label, SIGNAL(new_zoomed_parallel_view(int)), this, SLOT(emit_new_zoomed_parallel_view(int)));

	update_axis_label_info();
	set_min_max_visible(false);
}

PVParallelView::PVAxisGraphicsItem::~PVAxisGraphicsItem()
{
	if (scene()) {
		scene()->removeItem(this);
	}
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::paint
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::paint(QPainter *painter,
                                               const QStyleOptionGraphicsItem *option,
                                               QWidget *widget)
{
	painter->fillRect(
		0,
		-axis_extend,
	    PVParallelView::AxisWidth,
	    _axis_length + (2 * axis_extend),
	    lib_axis()->get_color().toQColor()
	);

	QGraphicsItemGroup::paint(painter, option, widget);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::update_axis_info
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::update_axis_label_info()
{
	_label->set_text(lib_axis()->get_name());
	_label->set_color(lib_axis()->get_titlecolor().toQColor());
	_label->set_axis_index(_lib_view.get_axes_combination().get_index_by_id(_axis_id));

	update_axis_min_max_info();
	update_layer_min_max_info();
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::update_axis_position
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::update_axis_label_position(const bool visible)
{
	if (visible) {
		_label->setPos(0, - 6 * axis_extend);
	} else {
		_label->setPos(0, - 2 * axis_extend);
	}
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::update_axis_min_max_info
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::update_axis_min_max_info()
{
	const Picviz::PVMapping *mapping = _lib_view.get_parent<Picviz::PVMapped>()->get_mapping();

	if (mapping == nullptr) {
		return;
	}

	const PVCol cur_axis = _lib_view.axes_combination.get_axis_column_index(_lib_view.axes_combination.get_index_by_id(_axis_id));
	const Picviz::mandatory_param_map &mand_params = mapping->get_mandatory_params_for_col(cur_axis);
	Picviz::mandatory_param_map::const_iterator it_min = mand_params.find(Picviz::mandatory_ymin);
	Picviz::mandatory_param_map::const_iterator it_max = mand_params.find(Picviz::mandatory_ymax);

	if (it_min == mand_params.end() || it_max == mand_params.end()) {
		PVLOG_WARN("ymin and/or ymax don't exist for axis %d. Maybe the mandatory minmax mapping hasn't be runned ?\n", cur_axis);
		return;
	}

	QColor col = lib_axis()->get_titlecolor().toQColor();
	// 42 because the tooltip has margins...
	int tooltip_width = QApplication::desktop()->screenGeometry().width() - 42;

	QString tmin = (*it_min).second.first;
	if (tmin.isEmpty()) {
		tmin = QString("(empty string)");
	}

	QString etmin, ttmin;
	make_min_max_text(_axis_min_value->font(), tmin, etmin, ttmin, _zone_width, tooltip_width);
	_axis_min_value->setPlainText(etmin);
	_axis_min_value->setToolTip(ttmin);
	_axis_min_value->setDefaultTextColor(col);

	QString tmax = (*it_max).second.first;
	if (tmax.isEmpty()) {
		tmax = QString("(empty string)");
	}

	QString etmax, ttmax;
	make_min_max_text(_axis_max_value->font(), tmax, etmax, ttmax, _zone_width, tooltip_width);
	_axis_max_value->setPlainText(etmax);
	_axis_max_value->setToolTip(ttmax);
	_axis_max_value->setDefaultTextColor(col);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::update_axis_min_max_position
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::update_axis_min_max_position()
{
	_axis_min_value->setPos(0, _axis_length + axis_extend);
	_axis_max_value->setPos(0, - 5 * axis_extend);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::update_layer_min_max_info
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::update_layer_min_max_info()
{
	const Picviz::PVMapping *mapping = _lib_view.get_parent<Picviz::PVMapped>()->get_mapping();

	if (mapping == nullptr) {
		return;
	}

	const PVCol cur_axis = _lib_view.axes_combination.get_axis_column_index(_lib_view.axes_combination.get_index_by_id(_axis_id));

	// 42 because the tooltip has margins...
	int tooltip_width = QApplication::desktop()->screenGeometry().width() - 42;

	const Picviz::PVLayer::list_row_indexes_t &vmins = _lib_view.get_current_layer().get_mins();
	const Picviz::PVLayer::list_row_indexes_t &vmaxs = _lib_view.get_current_layer().get_maxs();

	QString tmin;

	if ((size_t)cur_axis < vmins.size()) {
		tmin = _lib_view.get_data(vmins[cur_axis], cur_axis);
	}

	QString etmin, ttmin;
	make_min_max_text(_layer_min_value->font(), tmin, etmin, ttmin, _zone_width, tooltip_width);
	_layer_min_value->setPlainText(etmin);
	_layer_min_value->setToolTip(ttmin);
	_layer_min_value->setDefaultTextColor(Qt::white);

	QString tmax;

	if ((size_t)cur_axis < vmaxs.size()) {
		tmax = _lib_view.get_data(vmaxs[cur_axis], cur_axis);
	}

	QString etmax, ttmax;
	make_min_max_text(_layer_max_value->font(), tmax, etmax, ttmax, _zone_width, tooltip_width);
	_layer_max_value->setPlainText(etmax);
	_layer_max_value->setToolTip(ttmax);
	_layer_max_value->setDefaultTextColor(Qt::white);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::update_layer_min_max_position
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::update_layer_min_max_position()
{
	_layer_min_value->setPos(0, _axis_length + 3 * axis_extend);
	_layer_max_value->setPos(0, - 3 * axis_extend);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::set_min_max_visible
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::set_min_max_visible(const bool visible)
{
	_axis_min_value->setVisible(visible);
	_axis_max_value->setVisible(visible);
	_layer_min_value->setVisible(visible);
	_layer_max_value->setVisible(visible);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::get_label_scene_bbox
 *****************************************************************************/

QRectF PVParallelView::PVAxisGraphicsItem::get_label_scene_bbox() const
{
	return _label->get_scene_bbox();
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::lib_axis
 *****************************************************************************/

Picviz::PVAxis const* PVParallelView::PVAxisGraphicsItem::lib_axis() const
{
	return &_lib_view.get_axis_by_id(_axis_id);
}
