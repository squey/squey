//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <iostream>

#include <pvkernel/core/squey_bench.h> // for BENCH_END, BENCH_START
#include <pvkernel/widgets/PVUtils.h>

#include <squey/PVAxis.h>
#include <squey/PVView.h>
#include <squey/PVSource.h>

#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVAxisLabel.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVAxisHeader.h>
#include <pvparallelview/PVHitGraphBlocksManager.h>

#include <QApplication>
#include <QPainter>
#include <QGraphicsScene>
#include <QTimer>
#include <QToolTip>
#include <QDebug>

#define PROPERTY_TOOLTIP_VALUE "squey_property_tooltip"

static inline QString make_elided_text(const QFont& font, const QString& text, int elided_width)
{
	return QFontMetrics(font).elidedText(text, Qt::ElideRight, elided_width);
}

static void
set_item_text_value(QGraphicsTextItem* text_item, QString text, QColor const& color, int width)
{
	if (text.isEmpty()) {
		text = QObject::tr("(empty string)");
	}

	QString elided_txt(make_elided_text(text_item->font(), text, width));
	text_item->setPlainText(elided_txt);
	text_item->setProperty(PROPERTY_TOOLTIP_VALUE, text);
	text_item->setDefaultTextColor(color);
}

namespace PVParallelView::__impl
{

class PVToolTipEventFilter : public QObject
{
  public:
	explicit PVToolTipEventFilter(PVAxisGraphicsItem* parent) : QObject(parent) {}

  protected:
	bool eventFilter(QObject* obj, QEvent* ev) override
	{
		auto* gti = qobject_cast<QGraphicsTextItem*>(obj);
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
} // namespace PVParallelView

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem
 *****************************************************************************/

PVParallelView::PVAxisGraphicsItem::PVAxisGraphicsItem(PVParallelView::PVSlidersManager* sm_p,
                                                       Squey::PVView const& view,
                                                       PVCombCol comb_col,
                                                       PVRush::PVAxisFormat const& axis_fmt)
    : _sliders_manager_p(sm_p)
    , _comb_col(comb_col)
    , _axis_fmt(axis_fmt)
    , _lib_view(view)
    , _axis_length(10)
{
	setAcceptHoverEvents(true);   // This is needed to enable hover events
	setHandlesChildEvents(false); // This is needed to let the children of the
	                              // group handle their events.

	_event_filter = new __impl::PVToolTipEventFilter(this);

	// the sliders must be over all other QGraphicsItems
	setZValue(1.e42);

	_sliders_group = new PVSlidersGroup(sm_p, _comb_col, this);

	addToGroup(get_sliders_group());

	_label = new PVAxisLabel(view);
	addToGroup(_label);
	_label->setRotation(label_rotation);
	_label->setPos(0, -6 * axis_extend);

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

	set_min_max_visible(false);

	update_axis_label_info();
	update_axis_label_position();
	update_axis_min_max_position();
	update_layer_min_max_position();

	_header_zone = new PVAxisHeader(view, comb_col, this);
	addToGroup(_header_zone);
	connect(_header_zone, &PVAxisHeader::new_selection_slider,
	        [this]() { _sliders_group->add_selection_sliders(0, 1024); });
	connect(_header_zone, &PVAxisHeader::mouse_hover_entered, this,
	        &PVAxisGraphicsItem::mouse_hover_entered);
	connect(_header_zone, &PVAxisHeader::mouse_clicked, this, &PVAxisGraphicsItem::mouse_clicked);
	connect(_header_zone,
	        static_cast<void (PVAxisHeader::*)(PVCombCol)>(&PVAxisHeader::new_zoomed_parallel_view),
	        this, &PVAxisGraphicsItem::new_zoomed_parallel_view);
	connect(this, &PVAxisGraphicsItem::density_changed, [&]() { update(boundingRect());});
}

PVParallelView::PVAxisGraphicsItem::~PVAxisGraphicsItem()
{
	if (_axis_density_worker.joinable()) {
		_axis_density_worker_canceled.clear();
		_axis_density_worker.join();
	}

	if (scene()) {
		scene()->removeItem(this);
	}
}

QRectF PVParallelView::PVAxisGraphicsItem::boundingRect() const
{
	// The geometry of the axis changes whether min/max values are displayed or
	// not !
	// WARNING: if one graphics item is added to the axis group, its geometry must
	// be added here !!

	QRectF ret = _label->mapToParent(_label->boundingRect()).boundingRect();
	if (show_min_max_values()) {
		ret |= _axis_min_value->mapToParent(_axis_min_value->boundingRect()).boundingRect() |
		       _axis_max_value->mapToParent(_axis_max_value->boundingRect()).boundingRect() |
		       _layer_min_value->mapToParent(_layer_min_value->boundingRect()).boundingRect() |
		       _layer_max_value->mapToParent(_layer_max_value->boundingRect()).boundingRect();
	} else {
		int new_bottom = ret.bottom() + _axis_length + 2 * axis_extend;
		ret.setBottom(new_bottom);
	}

	return ret;
	// return childrenBoundingRect();
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::paint
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::paint(QPainter* painter,
                                               const QStyleOptionGraphicsItem* option,
                                               QWidget* widget)
{
	pvcop::db::INVALID_TYPE invalid =
	    _lib_view.get_parent<Squey::PVSource>().has_invalid(get_original_axis_column());

	if (not invalid) {
		painter->fillRect(0, -axis_extend, _axis_width, _axis_length + (2 * axis_extend),
		                  _axis_fmt.get_color().toQColor());
		if (_axis_density_enabled) {
			painter->drawImage(QRect{0, 0, int(_axis_width), int(_axis_length)},
			                   get_axis_density());
		}
	} else {
		const double valid_range = (1 - Squey::PVScalingFilter::INVALID_RESERVED_PERCENT_RANGE);

		painter->fillRect(0, -axis_extend, _axis_width,
		                  ((_axis_length * valid_range) + (axis_extend) + 2),
		                  _axis_fmt.get_color().toQColor());
		if (_axis_density_enabled) {
			painter->drawImage(QRect{0, 0, int(_axis_width), int(_axis_length)},
			                   get_axis_density());
		}

		// draw a circle for invalid/empty values
		if (invalid == pvcop::db::INVALID_TYPE::EMPTY) {
			painter->setBrush(_axis_fmt.get_color().toQColor());
		} else {
			painter->setBrush(Qt::black);
			painter->setPen(QPen(_axis_fmt.get_color().toQColor(), 1));
		}

		painter->drawEllipse(QPoint(1, _axis_length - 1), 3, 3);
	}

#ifdef SQUEY_DEVELOPER_MODE
	if (common::show_bboxes()) {
		painter->setPen(QPen(QColor(0, 0xFF, 0), 0));
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
	_label->set_text(_axis_fmt.get_name());
	_label->set_color(_axis_fmt.get_titlecolor().toQColor());

	update_axis_min_max_info();
	update_layer_min_max_info();
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::update_axis_position
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::update_axis_label_position()
{
	if (show_min_max_values()) {
		_label->setPos(0, -6 * axis_extend);
	} else {
		_label->setPos(0, -2 * axis_extend);
	}
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::update_axis_min_max_info
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::update_axis_min_max_info()
{
	const PVCombCol combined_col = get_combined_axis_column();

	const PVRow min_row = _lib_view.get_scaled_col_min_row(combined_col);
	const PVRow max_row = _lib_view.get_scaled_col_max_row(combined_col);

	set_axis_text_value(_axis_min_value, min_row);
	set_axis_text_value(_axis_max_value, max_row);
}

void PVParallelView::PVAxisGraphicsItem::set_axis_text_value(QGraphicsTextItem* item, PVRow const r)
{
	const PVCombCol combined_col = get_combined_axis_column();
	const QColor color = _axis_fmt.get_titlecolor().toQColor();
	const QString txt = QString::fromStdString(_lib_view.get_data(r, combined_col));

	set_item_text_value(item, txt, color, _zone_width);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::update_axis_min_max_position
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::update_axis_min_max_position()
{
	_axis_min_value->setPos(0, _axis_length + axis_extend);
	_axis_max_value->setPos(0, -5 * axis_extend);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::update_layer_min_max_info
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::update_layer_min_max_info()
{
	const Squey::PVLayer::list_row_indexes_t& vmins = _lib_view.get_current_layer().get_mins();
	const Squey::PVLayer::list_row_indexes_t& vmaxs = _lib_view.get_current_layer().get_maxs();

	const PVCol original_col = get_original_axis_column();
	PVRow min_row;
	PVRow max_row;
	if ((size_t)original_col >= vmins.size() || (size_t)original_col >= vmaxs.size()) {
		// Min/max values haven't been computed ! Take them from the scaled.
		const PVCombCol combined_col = get_combined_axis_column();
		min_row = _lib_view.get_scaled_col_min_row(combined_col);
		max_row = _lib_view.get_scaled_col_max_row(combined_col);
	} else {
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
	_header_zone->start(start);
}

/*****************************************************************************
 * PVParallelView::PVAxisGraphicsItem::update_layer_min_max_position
 *****************************************************************************/

void PVParallelView::PVAxisGraphicsItem::update_layer_min_max_position()
{
	_layer_min_value->setPos(0, _axis_length + 3 * axis_extend);
	_layer_max_value->setPos(0, -3 * axis_extend);
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
	} else {
		ret = mapToScene(QRectF(0, axis_extend, 0.1, axis_extend)).boundingRect();
	}
	return ret;
}

void PVParallelView::PVAxisGraphicsItem::show_tooltip(QGraphicsTextItem* gti,
                                                      QGraphicsSceneHelpEvent* event) const
{
	// Get tooltip original text
	QString text = gti->property(PROPERTY_TOOLTIP_VALUE).toString();

	// Word wrap it
	// 42 because the tooltip has margins...
	PVWidgets::PVUtils::html_word_wrap_text(text, gti->font(),
	                                        PVWidgets::PVUtils::tooltip_max_width(event->widget()));

	// And finally show this tooltip !
	QToolTip::showText(event->screenPos(), text, event->widget());
}

bool PVParallelView::PVAxisGraphicsItem::is_last_axis() const
{
	return _lib_view.get_axes_combination().is_last_axis(_comb_col);
}

void PVParallelView::PVAxisGraphicsItem::set_axis_length(uint32_t l)
{
	prepareGeometryChange();

	if (l != _axis_length) {
		refresh_density();
	}

	_axis_length = l;
	update_axis_label_position();
	update_axis_min_max_position();
	update_layer_min_max_position();
}

void PVParallelView::PVAxisGraphicsItem::set_zone_width(uint32_t zone_width, uint32_t axis_width)
{
	_zone_width = zone_width;
	_axis_width = axis_width;
	_header_zone->set_width(zone_width + axis_width);
	get_sliders_group()->setPos(axis_width / 2, 0.);
}

void PVParallelView::PVAxisGraphicsItem::enable_density(bool enable)
{
	_axis_density_enabled = enable;
	refresh_density();
}

void PVParallelView::PVAxisGraphicsItem::refresh_density()
{
	_axis_density_need_refresh = _axis_density_enabled;
}

void PVParallelView::PVAxisGraphicsItem::render_density(int axis_length)
{
	BENCH_START(render_density);

	std::vector<size_t> histogram(axis_length);

	auto const& scaled = _lib_view.get_parent<Squey::PVScaled>();
	auto col_data = scaled.get_column_pointer(get_original_axis_column());

	auto const& selection = _lib_view.get_real_output_selection();
	selection.visit_selected_lines([&histogram, col_data, axis_length](PVRow row) {
		size_t pixel_y = uint64_t(col_data[row]) * axis_length / (uint64_t(1) << 32);
		++histogram[axis_length - 1 - pixel_y];
	});

	if (not _axis_density_worker_canceled.test_and_set()) {
		return;
	}

	_axis_density_worker_result = QImage(1, axis_length, QImage::Format::Format_ARGB32);

	constexpr size_t density_spread = 3;
	const size_t histo_size = histogram.size();
	for (size_t i = 0; i < histo_size; ++i) {
		size_t sum = 0;
		for (size_t j = i >= density_spread ? i - density_spread : 0;
		     j < std::min(i + density_spread, histo_size); ++j) {
			sum += (density_spread - std::abs(int64_t(j) - int64_t(i))) * histogram[j];
		}
		_axis_density_worker_result.setPixelColor(
		    0, axis_length - 1 - i,
		    QColor::fromHsvF(std::max(0., 1. - double(sum) / double(selection.bit_count())) * 2. /
		                         3.,
		                     1, histogram[i] > 0, 1.));
	}

	BENCH_END(render_density, "render_density", _comb_col, 1, _comb_col, 1);
}

QImage PVParallelView::PVAxisGraphicsItem::get_axis_density()
{
	if (_axis_density_need_refresh) {
		_axis_density_need_refresh = false;
		if (not _axis_density_worker.joinable()) {
			_axis_density_worker_canceled.test_and_set();
			_axis_density_worker_finished.test_and_set();
			_axis_density_worker = std::thread([axis_length = _axis_length, this] {
				render_density(axis_length);
				_axis_density_worker_finished.clear();

				Q_EMIT density_changed();

			});
		} else {
			_axis_density_worker_canceled.clear();
		}
	}
	if (_axis_density_worker.joinable()) {
		if (not _axis_density_worker_finished.test_and_set()) {
			_axis_density_worker.join();
			if (not _axis_density_worker_result.isNull()) {
				_axis_density.swap(_axis_density_worker_result);
				_axis_density_worker_result = QImage();
			}
		}
	}
	return _axis_density;
}
