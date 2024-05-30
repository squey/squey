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

#ifndef PVPARALLELVIEW_PVAXISGRAPHICSITEM_H
#define PVPARALLELVIEW_PVAXISGRAPHICSITEM_H

#include <iostream>
#include <vector>
#include <utility>
#include <thread>

#include <QGraphicsItem>
class QPropertyAnimation;

#include <pvkernel/core/PVAlgorithms.h>

#include <squey/PVAxis.h>
#include <squey/PVView.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVSlidersManager.h>
#include <pvparallelview/PVSlidersGroup.h>

class QGraphicsSimpleTextItem;

namespace PVParallelView
{

class PVAxisHeader;

namespace __impl
{
class PVToolTipEventFilter;
class PVAxisSelectedAnimation;
} // namespace __impl

class PVAxisLabel;
class PVFullParallelScene;

class PVAxisGraphicsItem : public QObject, public QGraphicsItemGroup
{
	Q_OBJECT

	friend class __impl::PVToolTipEventFilter;
	friend class __impl::PVAxisSelectedAnimation;
	friend class PVAxisHeader;

  public:
	static constexpr qreal label_rotation = -45.;

  public:
	typedef PVSlidersGroup::selection_ranges_t selection_ranges_t;

	// Used to draw the axis out of the image zone
	constexpr static int axis_extend = 8;

  public:
	PVAxisGraphicsItem(PVSlidersManager* sm_p,
	                   Squey::PVView const& view,
	                   PVCombCol comb_col,
	                   PVRush::PVAxisFormat const& axis_fmt);
	~PVAxisGraphicsItem() override;

	void
	paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0) override;
	QRectF boundingRect() const override;

	void update_axis_label_info();
	void update_axis_min_max_info();
	void update_layer_min_max_info();

	void set_min_max_visible(const bool visible);

	PVSlidersGroup* get_sliders_group() { return _sliders_group; }

	const PVSlidersGroup* get_sliders_group() const { return _sliders_group; }

	PVCombCol get_combined_axis_column() const
	{
		// FIXME (pbrunet) : _comb_col is use only here to get data from PVView
		// but all PVView call convert back comb_col to nraw_col so we should only
		// need nraw_col stored in _axis_fmt.index
		return _comb_col;
	}

	PVCol get_original_axis_column() const { return _axis_fmt.index; }

	QString get_axis_type() const { return _axis_fmt.get_type(); }

	QRectF get_top_decoration_scene_bbox() const;
	QRectF get_bottom_decoration_scene_bbox() const;

	void set_axis_length(uint32_t l);

	void set_zone_width(uint32_t zone_width, uint32_t axis_width);

	QRect map_from_scene(QRectF rect) const
	{
		QPointF point = mapFromScene(rect.topLeft());
		return QRect(point.x(), point.y(), rect.width(), rect.height());
	}

	selection_ranges_t get_selection_ranges() const
	{
		return get_sliders_group()->get_selection_ranges();
	}

	bool is_last_axis() const;

	void highlight(bool start);

	PVAxisLabel* label() const { return _label; }

	QColor get_title_color() const { return _axis_fmt.get_titlecolor().toQColor(); }

	void enable_density(bool enable);
	void refresh_density();
	void render_density(int axis_length);
	QImage get_axis_density();

  public Q_SLOTS:
	void emit_new_zoomed_parallel_view(PVCombCol comb_col)
	{
		Q_EMIT new_zoomed_parallel_view(comb_col);
	}

  protected:
	void show_tooltip(QGraphicsTextItem* gti, QGraphicsSceneHelpEvent* event) const;

  Q_SIGNALS:
	void new_zoomed_parallel_view(PVCombCol comb_col);
	void mouse_hover_entered(PVCombCol axis, bool entered);
	void mouse_clicked(PVCombCol axis);
	void density_changed();

  private:
	void set_axis_text_value(QGraphicsTextItem* item, PVRow const r);
	inline bool show_min_max_values() const { return _minmax_visible; }

	void update_axis_label_position();
	void update_axis_min_max_position();
	void update_layer_min_max_position();

  private:
	PVSlidersManager* _sliders_manager_p;
	PVCombCol _comb_col;
	PVRush::PVAxisFormat const& _axis_fmt;
	QRectF _bbox;
	Squey::PVView const& _lib_view;
	PVSlidersGroup* _sliders_group;
	PVAxisLabel* _label;
	PVAxisHeader* _header_zone;
	uint32_t _axis_length;
	uint32_t _zone_width;
	uint32_t _axis_width;
	QGraphicsTextItem* _axis_min_value;
	QGraphicsTextItem* _axis_max_value;
	QGraphicsTextItem* _layer_min_value;
	QGraphicsTextItem* _layer_max_value;
	__impl::PVToolTipEventFilter* _event_filter;
	bool _minmax_visible;
	bool _axis_density_enabled = false;
	QImage _axis_density;
	bool _axis_density_need_refresh = false;
	std::thread _axis_density_worker;
	std::atomic_flag _axis_density_worker_canceled;
	std::atomic_flag _axis_density_worker_finished;
	QImage _axis_density_worker_result;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVAXISGRAPHICSITEM_H
