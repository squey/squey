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
#include <pvparallelview/PVParallelView.h>

#include <QApplication>
#include <QDesktopWidget>
#include <QPainter>
#include <QGraphicsScene>
#include <QToolTip>
#include <QGraphicsBlurEffect>
#include <QPropertyAnimation>

#define PROPERTY_TOOLTIP_VALUE "picviz_property_tooltip"

static inline QString make_elided_text(const QFont &font, const QString &text, int elided_width)
{
	return QFontMetrics(font).elidedText(text, Qt::ElideRight, elided_width);
}

static void set_item_text_value(QGraphicsTextItem* text_item, QString text, QColor const& color, int width)
{
	if (text.isEmpty()) {
		text = QObject::tr("(empty string)");
	}

	QString elided_txt(make_elided_text(text_item->font(), text, width));
	text_item->setPlainText(elided_txt);
	text_item->setProperty(PROPERTY_TOOLTIP_VALUE, text);
	text_item->setDefaultTextColor(color);
}


namespace PVParallelView { namespace __impl {

class PVToolTipEventFilter : public QObject
{
public:
	PVToolTipEventFilter(PVAxisGraphicsItem* parent):
		QObject(parent)
	{ }

protected:
	bool eventFilter(QObject *obj, QEvent *ev)
	{
		QGraphicsTextItem* gti = qobject_cast<QGraphicsTextItem*>(obj);
		if (!gti) {
			return false;
		}

		switch (ev->type()) {
		case QEvent::GraphicsSceneHelp:
			agi_parent()->show_tooltip(gti, static_cast<QGraphicsSceneHelpEvent*>(ev));
			ev->accept();
			break;
		default:
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

} } // PVParallelView::__impl

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem
 *****************************************************************************/

PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem(PVParallelView::PVSlidersManager_p sm_p,
                                                       Picviz::PVView const& view, const axis_id_t &axis_id) :
	_sliders_manager_p(sm_p),
	_axis_id(axis_id),
	_lib_view(view),
	_axis_length(10),
	_axis_selected_animation(new __impl::PVAxisSelectedAnimation(this))
{
	setAcceptHoverEvents(true); // This is needed to enable hover events
	setHandlesChildEvents(false); // This is needed to let the children of the group handle their events.

	_event_filter = new __impl::PVToolTipEventFilter(this);

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
	connect(_label, SIGNAL(mouse_hover_entered(PVCol, bool)), this, SIGNAL(mouse_hover_entered(PVCol, bool)));
	connect(_label, SIGNAL(mouse_clicked(PVCol)), this, SIGNAL(mouse_clicked(PVCol)));

	set_min_max_visible(false);

	update_axis_label_info();
	update_axis_label_position();
	update_axis_min_max_position();
	update_layer_min_max_position();
}

PVParallelView::PVAxisGraphicsItem::~PVAxisGraphicsItem()
{
	if (scene()) {
		scene()->removeItem(this);
	}
}

QRectF PVParallelView::PVAxisGraphicsItem::boundingRect() const
{
	// The geometry of the axis changes whether min/max values are displayed or not !
	// WARNING: if one graphics item is added to the axis group, its geometry must be added here !!
	
	QRectF ret = _label->mapToParent(_label->boundingRect()).boundingRect();
	if (show_min_max_values()) {
		ret |= _axis_min_value->mapToParent(_axis_min_value->boundingRect()).boundingRect()   |
               _axis_max_value->mapToParent(_axis_max_value->boundingRect()).boundingRect()   |
               _layer_min_value->mapToParent(_layer_min_value->boundingRect()).boundingRect() |
               _layer_max_value->mapToParent(_layer_max_value->boundingRect()).boundingRect();
	}
	else {
		int new_bottom = ret.bottom() + _axis_length + 2*axis_extend;
		ret.setBottom(new_bottom);
	}

	return ret;
	//return childrenBoundingRect();
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

#ifdef PICVIZ_DEVELOPER_MODE
	if (common::show_bboxes()) {
		painter->setPen(QColor(0, 0xFF, 0));
		painter->drawRect(boundingRect());
	}
#endif

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

void PVParallelView::PVAxisGraphicsItem::update_axis_label_position()
{
	if (show_min_max_values()) {
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

	const PVCol combined_col = get_combined_axis_column();

	const PVRow min_row = _lib_view.get_plotted_col_min_row(combined_col);
	const PVRow max_row = _lib_view.get_plotted_col_max_row(combined_col);

	set_axis_text_value(_axis_min_value, min_row);
	set_axis_text_value(_axis_max_value, max_row);
}

void PVParallelView::PVAxisGraphicsItem::set_axis_text_value(QGraphicsTextItem* item, PVRow const r)
{
	const PVCol combined_col = get_combined_axis_column();
	const QColor color = lib_axis()->get_titlecolor().toQColor();
	const QString txt = _lib_view.get_data(r, combined_col);

	set_item_text_value(item, txt, color, _zone_width);
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

	const Picviz::PVLayer::list_row_indexes_t &vmins = _lib_view.get_current_layer().get_mins();
	const Picviz::PVLayer::list_row_indexes_t &vmaxs = _lib_view.get_current_layer().get_maxs();

	const PVCol original_col = get_original_axis_column();
	PVRow min_row;
	PVRow max_row;
	if ((size_t) original_col >= vmins.size() || (size_t) original_col >= vmaxs.size()) {
		// Min/max values haven't been computed ! Take them from the plotted.
		const PVCol combined_col = get_combined_axis_column();
		min_row = _lib_view.get_plotted_col_min_row(combined_col);
		max_row = _lib_view.get_plotted_col_max_row(combined_col);
	}
	else {
		min_row = vmins[original_col];
		max_row = vmaxs[original_col];
	}

	set_axis_text_value(_layer_min_value, min_row);
	set_axis_text_value(_layer_max_value, max_row);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::highlight
 *****************************************************************************/
void PVParallelView::PVAxisGraphicsItem::highlight(bool start)
{
	_axis_selected_animation->start(start);
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
	prepareGeometryChange();

	_minmax_visible = visible;
	_axis_min_value->setVisible(visible);
	_axis_max_value->setVisible(visible);
	_layer_min_value->setVisible(visible);
	_layer_max_value->setVisible(visible);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::get_label_scene_bbox
 *****************************************************************************/

QRectF PVParallelView::PVAxisGraphicsItem::get_top_decoration_scene_bbox() const
{
	QRectF ret = _label->get_scene_bbox();
	if (show_min_max_values()) {
		ret |= _axis_max_value->sceneBoundingRect() | _layer_max_value->sceneBoundingRect();
	}
	ret.setTop(ret.top() - axis_extend);
	return ret;
}

QRectF PVParallelView::PVAxisGraphicsItem::get_bottom_decoration_scene_bbox() const
{
	QRectF ret;
	if (show_min_max_values()) {
		ret = _axis_min_value->sceneBoundingRect() | _layer_min_value->sceneBoundingRect();
		ret.setBottom(ret.bottom() + axis_extend);
	}
	else {
		ret = mapToScene(QRectF(0, axis_extend, 0.1, axis_extend)).boundingRect();
	}
	return ret;
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::lib_axis
 *****************************************************************************/

Picviz::PVAxis const* PVParallelView::PVAxisGraphicsItem::lib_axis() const
{
	return &_lib_view.get_axis_by_id(_axis_id);
}

void PVParallelView::PVAxisGraphicsItem::show_tooltip(QGraphicsTextItem* gti, QGraphicsSceneHelpEvent* event) const
{
	// Get tooltip original text
	QString text = gti->property(PROPERTY_TOOLTIP_VALUE).toString();

	// Word wrap it
	// 42 because the tooltip has margins...
	PVWidgets::PVUtils::html_word_wrap_text(text, gti->font(), PVWidgets::PVUtils::tooltip_max_width(event->widget()));

	// And finally show this tooltip !
	QToolTip::showText(event->screenPos(), text, event->widget());
}

bool PVParallelView::PVAxisGraphicsItem::is_last_axis() const
{
	return _lib_view.is_last_axis(_axis_id);
}

void PVParallelView::PVAxisGraphicsItem::set_axis_length(int l)
{
	prepareGeometryChange();

	_axis_length = l;
	update_axis_label_position();
	update_axis_min_max_position();
	update_layer_min_max_position();
}

PVParallelView::__impl::PVAxisSelectedAnimation::PVAxisSelectedAnimation(PVAxisGraphicsItem* parent) :
	QObject(parent)
{
	// Selected axis symbol
	_selected_axis_hole = new QGraphicsPixmapItem(QPixmap(":/location_hole").scaled(QSize(50, 50), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	axis()->addToGroup(_selected_axis_hole);
	_selected_axis_hole->setPos(- axis()->_axis_length - (1.73 * 8), - 64);
	_selected_axis_hole->setVisible(false);
	_selected_axis_dot = new QGraphicsPixmapItem(QPixmap(":/location_dot").scaled(QSize(50, 50), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	axis()->addToGroup(_selected_axis_dot);
	_selected_axis_dot->setPos(- axis()->_axis_length - (1.73 * 8), - 64);
	_selected_axis_dot->setVisible(false);

	// Setup opacity animation
	_opacity_animation = new QPropertyAnimation(this, "opacity");
	_opacity_animation->setStartValue(0.0);
	_opacity_animation->setEndValue(1.0);
	_opacity_animation->setDuration(opacity_animation_duration_ms);
	_opacity_animation->setEasingCurve(QEasingCurve::InOutQuad);
	QGraphicsOpacityEffect* opacity_effect = new QGraphicsOpacityEffect();
	_selected_axis_hole->setGraphicsEffect(opacity_effect);

	// Setup blur animation
	_blur_animation = new QPropertyAnimation(this, "blur");
	_blur_animation->setStartValue(blur_animation_min_amount);
	_blur_animation->setEndValue(blur_animation_max_amount);
	_blur_animation->setDuration(blur_animation_duration_ms);
	_blur_animation->setEasingCurve(blur_animation_easing);
	connect(
		_blur_animation,
		SIGNAL(currentLoopChanged(int)),
		this,
		SLOT(blur_animation_current_loop_changed())
	);
	QGraphicsBlurEffect* blur_effect = new QGraphicsBlurEffect();
	_selected_axis_dot->setGraphicsEffect(blur_effect);
}

void PVParallelView::__impl::PVAxisSelectedAnimation::start(bool start)
{
	if (start) {
		_selected_axis_hole->setVisible(true);
		_opacity_animation->setDirection(QAbstractAnimation::Forward);
		_opacity_animation->start();

		_selected_axis_dot->setVisible(true);
		_blur_animation->setLoopCount(-1); // Forever and ever...
		_blur_animation->setDirection(QAbstractAnimation::Forward);
		_blur_animation->start();
	}
	else {
		_opacity_animation->setDirection(QAbstractAnimation::Backward);
		_opacity_animation->start();

		_blur_animation->stop();
		_selected_axis_dot->setVisible(false);
	}
}

void PVParallelView::__impl::PVAxisSelectedAnimation::blur_animation_current_loop_changed()
{
}

void PVParallelView::__impl::PVAxisSelectedAnimation::set_opacity(qreal opacity)
{
	QGraphicsOpacityEffect* opacity_effect = (QGraphicsOpacityEffect*) _selected_axis_hole->graphicsEffect();
	opacity_effect->setOpacity(opacity);
}

void PVParallelView::__impl::PVAxisSelectedAnimation::set_blur(qreal blur)
{
	QGraphicsBlurEffect* blur_effect = (QGraphicsBlurEffect*) _selected_axis_dot->graphicsEffect();
	blur_effect->setBlurRadius(blur);
}
