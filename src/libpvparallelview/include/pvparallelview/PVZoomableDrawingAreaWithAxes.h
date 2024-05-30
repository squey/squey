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

#ifndef PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAWITHAXES_H
#define PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAWITHAXES_H

#include <pvparallelview/PVZoomableDrawingArea.h>
#include <squey/widgets/PVAxisComboBox.h>

#include <QString>

class QWidget;

namespace PVParallelView
{

/**
 * @class PVZoomableDrawingAreaWithAxes
 *
 * A canvas to draw charts.
 *
 * @note As this widget use a QScrollArea and as all widgets use left-hand
 * coordinates system, this widget must follow this constraint. However, all
 * the internal computations use right-hand coordinates system to render
 * correctly the decoration and to interact with it. So that, your subclass
 * must make do:
 * - the scene must be defined as QRect(0, -N, M, N);
 * - your calculation must consider that ordinates are negative instead of
 *   being positive (doing a -v is generally enough to do the things right);
 * - the zoom's range and initial value must be properly set to have the
 *   expected viewport (for example, for [0,2^32]^2, the zoom range will be
 *   [-22, 0] for the default zoom behaviour);
 * - strictly positive value for the zoom range upper bound will permit to
 *   zoom further than theorical value precision.
 *
 * @note the scales are continuous, not discrete.
 *
 * @todo make sure that scene resizing properly works, for a chart like the
 * "hitgraph" view, changing ordinates range has not been tested.
 *
 * @todo axes actually hide "0" values. They also must be moved (one pixel to
 * the bottom for the abscissae and one pixel to the left for ordinates").
 *
 * @todo the ticks must be moving instead of being fixed
 *
 * @todo allow to specify per axis scale type : linear, logarithm
 *
 * @todo: allow to specify scale position
 *
 * @todo allow to have a context menu.
 */

class PVZoomableDrawingAreaWithAxes : public PVZoomableDrawingArea
{
  protected:
	// minimum gap (in pixel) between 2 subticks to make them visible of not.
	constexpr static int subtick_min_gap = 32;

	static constexpr int MAX_TEXT_LABEL_WIDTH =
	    120; /*!< The maximum width of a text label in pixel. This value should be calculated later,
depend of the client's windows settings. */

  public:
	/**
	 * CTOR
	 */
	explicit PVZoomableDrawingAreaWithAxes(QWidget* parent = nullptr);

	/**
	 * DTOR
	 */
	~PVZoomableDrawingAreaWithAxes() override;

	/**
	 * Set the color to use when drawing scales & legends
	 *
	 * @param color the color to use
	 */
	void set_decoration_color(const QColor& color);

	/**
	 * Return the color used when drawing scales & legends
	 *
	 */
	QColor get_decoration_color() const { return _decoration_color; }

	/**
	 * Set the legend for the horizontal axis
	 *
	 * @param legend the text to use
	 */
	void set_x_legend(const QString& legend);

	void set_x_legend(PVWidgets::PVAxisComboBox* legend);

	/**
	 * Return the legend of the horizontal axis
	 */
	const QString& get_x_legend() const { return _x_legend; }

	/**
	 * Set the legend for the vertical axis
	 *
	 * @param legend the text to use
	 */
	void set_y_legend(const QString& legend);

	void set_y_legend(PVWidgets::PVAxisComboBox* legend);

	/**
	 * Return the legend of the vertical axis
	 */
	const QString& get_y_legend() const { return _y_legend; }

	/**
	 * Set the ticks count per level
	 */
	void set_ticks_per_level(int n);

	/**
	 * Return the current ticks count
	 */
	int get_ticks_per_level() const { return _ticks_per_level; }

  protected:
	/**
	 * Return the text to print in x scale for a given value
	 *
	 * This method has to be overloaded in a subclass to display
	 * information related to the view.
	 *
	 * @param value the value for which we want a text
	 *
	 * @return the label corresponding to value
	 */
	virtual QString get_x_value_at(const qint64 value);

	/**
	 * Return the text to print on the y scale for a given value
	 *
	 * This method has to be overloaded in a subclass to display
	 * information related to the view.
	 *
	 * @param value the value for which we want a text
	 *
	 * @return the label corresponding to value
	 */
	virtual QString get_y_value_at(const qint64 value);

	int get_x_axis_length() const { return _x_axis_length; }

	int get_y_axis_length() const { return _y_axis_length; }

	QString get_elided_text(const QString& text) const;

  protected:
	void recompute_margins() override;
	void recompute_decorations();
	virtual void draw_decorations(QPainter* painter, const QRectF& rect);

  protected:
	void drawBackground(QPainter* painter, const QRectF& rect) override;

  private:
	void draw_deco_v3(QPainter* painter, const QRectF& rect);

  private:
	QColor _decoration_color;
	QString _x_legend;
	QString _y_legend;
	PVWidgets::PVAxisComboBox* _x_legend_widget = nullptr;
	PVWidgets::PVAxisComboBox* _y_legend_widget = nullptr;

	int _ticks_per_level;

	int _x_axis_length;
	int _y_axis_length;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAWITHAXES_H
