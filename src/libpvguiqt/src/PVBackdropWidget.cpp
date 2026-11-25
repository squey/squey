//
// MIT License
//
// © Squey, 2026
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

#include <pvguiqt/PVBackdropWidget.h>

#include <QEvent>
#include <QGuiApplication>
#include <QLinearGradient>
#include <QPaintEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPolygonF>
#include <QRadialGradient>
#include <QRegion>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <numbers>
#include <optional>
#include <random>
#include <utility>

namespace
{

/** As the theme rounds the corners of cards, in pixels. */
constexpr int CARD_RADIUS = 8;
/** Between the axes of the drawing, in pixels. */
constexpr int AXIS_SPACING = 160;
/**
 * How long, in milliseconds, the selection takes to pass from a group of rows to the next, and the
 * light to sweep over the widget.
 */
constexpr int FADE_MS = 1600;
constexpr int SWEEP_MS = 12000;
/** Between two frames, in milliseconds: a color fading needs fewer than a light moving. */
constexpr int FADE_FRAME_MS = 80;
constexpr int SWEEP_FRAME_MS = 50;
/**
 * Of the band the light covers, in pixels, and the stops of the gradient that brightens it towards
 * its middle.
 */
constexpr int GLOW_WIDTH = 280;
constexpr int GLOW_STOPS = 16;
/** Around the rows of a group, in pixels: for the halo of the light, and the antialiasing. */
constexpr int ROWS_MARGIN = 4;
/** Beyond the edges of a clearing, on each side, in pixels: where the drawing fades in. */
constexpr int CLEARING_WIDTH = 128;
constexpr int CLEARING_HEIGHT = 32;

/** Rows made up for the drawing: groups of close rows, and a few rows of none. */
struct Rows {
	static constexpr int AXES = 32;
	static constexpr int GROUPS = 6;
	/** The group of the rows of none, which is never selected. */
	static constexpr int NO_GROUP = -1;
	static constexpr int FIRST_SELECTED = 2;

	struct Row {
		std::array<float, AXES> values;
		int group;
	};

	Rows()
	{
		// The same drawing in every page.
		std::mt19937 random(5);
		std::uniform_real_distribution<float> uniform(0.08f, 0.92f);
		std::normal_distribution<float> step(0.f, 0.16f);
		for (int group = 0; group < GROUPS; group++) {
			std::array<float, AXES> profile;
			profile[0] = uniform(random);
			for (int axis = 1; axis < AXES; axis++) {
				profile[axis] = std::clamp(profile[axis - 1] + step(random), 0.05f, 0.95f);
			}
			std::normal_distribution<float> spread(0.f, 0.015f + 0.008f * float(group));
			for (int row = 0; row < 12 + 4 * group; row++) {
				Row& added = rows.emplace_back(Row{.values = {}, .group = group});
				for (int axis = 0; axis < AXES; axis++) {
					added.values[axis] = std::clamp(profile[axis] + spread(random), 0.f, 1.f);
				}
			}
		}
		for (int row = 0; row < 12; row++) {
			Row& outlier = rows.emplace_back(Row{.values = {}, .group = NO_GROUP});
			for (float& value : outlier.values) {
				value = uniform(random);
			}
		}
	}

	std::vector<Row> rows;
};

/** Where the axes of the drawing of a widget of a size go. */
struct Plot {
	explicit Plot(QSize size)
	    // Axes from one edge to the other, as many on each side of the center.
	    : axes(std::min(Rows::AXES, size.width() / AXIS_SPACING + 2))
	    , first((size.width() - (axes - 1) * AXIS_SPACING) / 2.)
	    , top(size.height() * 0.06)
	    , height(size.height() * 0.88)
	{
	}

	qreal x_of(int axis) const { return first + axis * AXIS_SPACING; }
	qreal y_of(float value) const { return top + height * (1.f - value); }

	/** Draws with @p pen the rows of @p group, all of them without one. */
	void draw_rows(QPainter& painter, QPen const& pen, std::optional<int> group = {}) const
	{
		painter.setPen(pen);
		for (const Rows::Row& row : made_up().rows) {
			if (group and row.group != *group) {
				continue;
			}
			QPolygonF line;
			for (int axis = 0; axis < axes; axis++) {
				line << QPointF(x_of(axis), y_of(row.values[axis]));
			}
			painter.drawPolyline(line);
		}
	}

	/**
	 * Where the rows of @p group go, and @p margin around: from one axis to the next, in slices
	 * narrow enough for the rows to fill much of them as they rise or fall.
	 */
	QRegion area_of(int group, int margin) const
	{
		constexpr int SLICES = 4;
		QRegion area;
		for (int axis = 0; axis + 1 < axes; axis++) {
			for (int slice = 0; slice < SLICES; slice++) {
				const float from = float(slice) / SLICES;
				const float to = float(slice + 1) / SLICES;
				float lowest = 1;
				float highest = 0;
				for (const Rows::Row& row : made_up().rows) {
					if (row.group == group) {
						const float left = std::lerp(row.values[axis], row.values[axis + 1], from);
						const float right = std::lerp(row.values[axis], row.values[axis + 1], to);
						lowest = std::min({lowest, left, right});
						highest = std::max({highest, left, right});
					}
				}
				area |= QRectF(QPointF(x_of(axis) + AXIS_SPACING * from, y_of(highest)),
				               QPointF(x_of(axis) + AXIS_SPACING * to, y_of(lowest)))
				            .toAlignedRect()
				            .adjusted(-margin, -margin, margin, margin);
			}
		}
		return area;
	}

	static Rows const& made_up()
	{
		static const Rows rows;
		return rows;
	}

	int axes;
	qreal first;
	qreal top;
	qreal height;
};

/** A transparent pixmap of @p size, for a screen of @p ratio. */
QPixmap transparent_pixmap(QSize size, qreal ratio)
{
	QPixmap pixmap(size * ratio);
	pixmap.setDevicePixelRatio(ratio);
	pixmap.fill(Qt::transparent);
	return pixmap;
}

QColor with_alpha(QColor color, float alpha)
{
	color.setAlphaF(alpha);
	return color;
}

/**
 * The drawing of a widget of @p size, for a screen of @p ratio, but its selection: axes and rows
 * in @p color.
 */
QPixmap decoration(QSize size, qreal ratio, QColor const& color)
{
	QPixmap pixmap = transparent_pixmap(size, ratio);
	QPainter painter(&pixmap);
	painter.setRenderHint(QPainter::Antialiasing);
	const Plot plot(size);
	painter.setPen(QPen(with_alpha(color, 0.12f), 1));
	for (int axis = 0; axis < plot.axes; axis++) {
		painter.drawLine(QPointF(plot.x_of(axis), plot.top),
		                 QPointF(plot.x_of(axis), plot.top + plot.height));
	}
	plot.draw_rows(painter, QPen(with_alpha(color, 0.045f), 1.2));
	painter.end();
	return pixmap;
}

/**
 * The rows of @p group in the drawing of a widget of @p size, for a screen of @p ratio, drawn with
 * each of @p pens in turn.
 */
QPixmap rows_of(QSize size, qreal ratio, int group, std::initializer_list<QPen> pens)
{
	QPixmap pixmap = transparent_pixmap(size, ratio);
	QPainter painter(&pixmap);
	painter.setRenderHint(QPainter::Antialiasing);
	const Plot plot(size);
	for (QPen const& pen : pens) {
		plot.draw_rows(painter, pen, group);
	}
	painter.end();
	return pixmap;
}

/** The rows of @p group as Squey draws a selection in @p color. */
QPixmap selection(QSize size, qreal ratio, int group, QColor const& color)
{
	return rows_of(size, ratio, group, {QPen(with_alpha(color, 0.28f), 1.2)});
}

/** The rows of @p group selected in @p color as they light up when the light goes over them. */
QPixmap glow(QSize size, qreal ratio, int group, QColor const& color)
{
	return rows_of(
	    size, ratio, group,
	    {QPen(with_alpha(color, 0.1f), 5), QPen(with_alpha(color.lighter(130), 0.8f), 1.4)});
}

/**
 * The part @p lit of @p glow, as the light of @p band shows it: brighter towards the middle of the
 * band. For a screen of @p ratio.
 */
QPixmap in_light(QPixmap const& glow, QRect const& band, QRect const& lit, qreal ratio)
{
	QPixmap pixmap = transparent_pixmap(lit.size(), ratio);
	QPainter painter(&pixmap);
	painter.drawPixmap(-lit.topLeft(), glow);
	QLinearGradient brightness(band.left() - lit.left(), 0, band.left() + band.width() - lit.left(),
	                           0);
	for (int stop = 0; stop <= GLOW_STOPS; stop++) {
		const double alpha = std::sin(std::numbers::pi * stop / GLOW_STOPS);
		brightness.setColorAt(double(stop) / GLOW_STOPS, with_alpha(Qt::black, float(alpha)));
	}
	painter.setCompositionMode(QPainter::CompositionMode_DestinationIn);
	painter.fillRect(QRect(QPoint(0, 0), lit.size()), brightness);
	painter.end();
	return pixmap;
}

} // namespace

PVGuiQt::PVBackdropWidget::PVBackdropWidget(QWidget* parent)
    : QWidget(parent)
    , _decoration_color(palette().color(QPalette::Highlight))
    , _selected(Rows::FIRST_SELECTED)
    , _next(_selected)
{
	_fade_timer.setInterval(FADE_FRAME_MS);
	connect(&_fade_timer, &QTimer::timeout, this, &PVBackdropWidget::advance_fade);
	_sweep_timer.setInterval(SWEEP_FRAME_MS);
	connect(&_sweep_timer, &QTimer::timeout, this, &PVBackdropWidget::advance_sweep);
	connect(qGuiApp, &QGuiApplication::applicationStateChanged, this,
	        &PVBackdropWidget::update_animation);
}

void PVGuiQt::PVBackdropWidget::set_decoration_color(QColor const& color)
{
	_decoration_color = color;
	_selection = {};
	_next_selection = {};
	_glow = {};
	update();
}

void PVGuiQt::PVBackdropWidget::add_card(QWidget* card, Card kind)
{
	_cards.emplace_back(card);
	if (kind == Card::Opaque) {
		_opaque_cards.emplace_back(card);
	}
	follow(card);
}

void PVGuiQt::PVBackdropWidget::add_clearing(QWidget* widget)
{
	_clearings.emplace_back(widget);
	follow(widget);
}

void PVGuiQt::PVBackdropWidget::follow(QWidget* widget)
{
	for (QWidget* moving = widget; moving != nullptr and moving != this;
	     moving = moving->parentWidget()) {
		moving->installEventFilter(this);
	}
}

void PVGuiQt::PVBackdropWidget::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);
	painter.fillRect(rect(), palette().color(QPalette::Window));
	const qreal ratio = devicePixelRatioF();
	if (_decoration.size() != size() * ratio) {
		_decoration = decoration(size(), ratio, palette().color(QPalette::WindowText));
	}
	painter.drawPixmap(0, 0, _decoration);

	// The selection over the other rows, as Squey draws it, in a color fading into theirs as it
	// passes to the next group. The rows are drawn once: Qt strokes long antialiased lines of this
	// width slowly, a millisecond each.
	if (_selection.size() != _decoration.size()) {
		_selection = selection(size(), ratio, _selected, _decoration_color);
	}
	painter.setOpacity(1 - _fade);
	painter.drawPixmap(0, 0, _selection);
	if (_next != _selected) {
		if (_next_selection.size() != _decoration.size()) {
			_next_selection = selection(size(), ratio, _next, _decoration_color);
		}
		painter.setOpacity(_fade);
		painter.drawPixmap(0, 0, _next_selection);
	}
	painter.setOpacity(1);
	// The light goes over the lit rows in a pixmap of their own, painted at once: each painting
	// through the ragged region repainted costs at each of its lines.
	if (const QRect lit = _glow_band & event->region().boundingRect(); not lit.isEmpty()) {
		if (_glow.size() != _decoration.size()) {
			_glow = glow(size(), ratio, _selected, _decoration_color);
		}
		painter.drawPixmap(lit.topLeft(), in_light(_glow, _glow_band, lit, ratio));
	}

	// What shows on the background alone, in an ellipse of it fading out beyond its edges.
	const QColor background = palette().color(QPalette::Window);
	for (const QPointer<QWidget>& clearing : _clearings) {
		if (clearing == nullptr or not clearing->isVisible()) {
			continue;
		}
		QRadialGradient fading(0.5, 0.5, 0.5);
		fading.setCoordinateMode(QGradient::ObjectMode);
		fading.setColorAt(0, background);
		fading.setColorAt(0.6, background);
		fading.setColorAt(1, with_alpha(background, 0));
		painter.fillRect(
		    QRect(clearing->mapTo(this, QPoint(0, 0)), clearing->size())
		        .adjusted(-CLEARING_WIDTH, -CLEARING_HEIGHT, CLEARING_WIDTH, CLEARING_HEIGHT),
		    fading);
	}

	// A shadow, wider and fainter from each card, a bit below it. Not under it, where it would
	// darken a card the drawing shows through: each layer reaches at least as high as the card,
	// which makes a hole in it.
	painter.setRenderHint(QPainter::Antialiasing);
	painter.setPen(Qt::NoPen);
	painter.setBrush(QColor(0, 0, 0, 7));
	for (const QPointer<QWidget>& card : _cards) {
		if (card == nullptr or not card->isVisible()) {
			continue;
		}
		const QRectF area(card->mapTo(this, QPoint(0, 0)), card->size());
		for (int spread = 1; spread <= 8; spread++) {
			QPainterPath layer;
			layer.setFillRule(Qt::OddEvenFill);
			layer.addRoundedRect(
			    area.adjusted(-spread, std::min(0, 2 - spread), spread, 2 + spread),
			    CARD_RADIUS + spread, CARD_RADIUS + spread);
			layer.addRoundedRect(area, CARD_RADIUS, CARD_RADIUS);
			painter.drawPath(layer);
		}
	}
}

void PVGuiQt::PVBackdropWidget::changeEvent(QEvent* event)
{
	// The colors of a theme, which polishes the widget anew.
	if (event->type() == QEvent::PaletteChange or event->type() == QEvent::StyleChange) {
		_decoration = {};
		update();
	}
	QWidget::changeEvent(event);
}

void PVGuiQt::PVBackdropWidget::showEvent(QShowEvent* event)
{
	QWidget::showEvent(event);
	update_animation();
}

void PVGuiQt::PVBackdropWidget::hideEvent(QHideEvent* event)
{
	QWidget::hideEvent(event);
	update_animation();
}

void PVGuiQt::PVBackdropWidget::update_animation()
{
	if (not isVisible() or QGuiApplication::applicationState() != Qt::ApplicationActive) {
		if (_fade_timer.isActive()) {
			end_fade();
		}
		if (_sweep_timer.isActive()) {
			end_sweep();
		}
	} else if (not _fade_timer.isActive() and not _sweep_timer.isActive()) {
		start_sweep();
	}
}

void PVGuiQt::PVBackdropWidget::start_fade()
{
	_next = (_selected + 1) % Rows::GROUPS;
	_clock.start();
	_fade_timer.start();
}

void PVGuiQt::PVBackdropWidget::advance_fade()
{
	const qreal progress = _clock.elapsed() / qreal(FADE_MS);
	if (progress >= 1) {
		end_fade();
		start_sweep();
		return;
	}
	// Slow at both ends.
	_fade = progress * progress * (3 - 2 * progress);
	update_selection();
}

void PVGuiQt::PVBackdropWidget::end_fade()
{
	_fade_timer.stop();
	update_selection();
	_selected = _next;
	_selection = std::move(_next_selection);
	_next_selection = {};
	_fade = 0;
}

void PVGuiQt::PVBackdropWidget::start_sweep()
{
	_clock.start();
	_sweep_timer.start();
}

void PVGuiQt::PVBackdropWidget::advance_sweep()
{
	const qint64 elapsed = _clock.elapsed();
	if (elapsed >= SWEEP_MS) {
		end_sweep();
		start_fade();
		return;
	}
	// From beyond the left edge to beyond the right one.
	const int left = int((width() + GLOW_WIDTH) * elapsed / SWEEP_MS) - GLOW_WIDTH;
	move_glow(QRect(left, 0, GLOW_WIDTH, height()));
}

void PVGuiQt::PVBackdropWidget::end_sweep()
{
	_sweep_timer.stop();
	move_glow({});
	// The next sweep lights up the rows of another group.
	_glow = {};
}

void PVGuiQt::PVBackdropWidget::update_selection()
{
	// Where the rows of the two groups go: nothing changes elsewhere.
	const Plot plot(size());
	update_drawing(plot.area_of(_selected, ROWS_MARGIN) | plot.area_of(_next, ROWS_MARGIN));
}

void PVGuiQt::PVBackdropWidget::move_glow(QRect const& band)
{
	// Where the light goes over the selected rows: there is nothing to light up elsewhere.
	update_drawing((QRegion(_glow_band) | QRegion(band)) &
	               Plot(size()).area_of(_selected, ROWS_MARGIN));
	_glow_band = band;
}

void PVGuiQt::PVBackdropWidget::update_drawing(QRegion area)
{
	// The drawing does not show through the opaque cards, which are not painted again but at
	// their rounded corners.
	for (const QPointer<QWidget>& card : _opaque_cards) {
		if (card != nullptr and card->isVisible()) {
			const QRect inside(card->mapTo(this, QPoint(0, 0)), card->size());
			area -= QRegion(inside.adjusted(CARD_RADIUS, 0, -CARD_RADIUS, 0)) |
			        QRegion(inside.adjusted(0, CARD_RADIUS, 0, -CARD_RADIUS));
		}
	}
	update(area);
}

bool PVGuiQt::PVBackdropWidget::eventFilter(QObject* watched, QEvent* event)
{
	const auto moves = [watched](std::vector<QPointer<QWidget>> const& followed) {
		return std::any_of(followed.begin(), followed.end(), [watched](QPointer<QWidget> const& w) {
			return w == watched or (w != nullptr and watched->isWidgetType() and
			                        static_cast<QWidget*>(watched)->isAncestorOf(w));
		});
	};
	switch (event->type()) {
	case QEvent::Move:
	case QEvent::Resize:
	case QEvent::Show:
	case QEvent::Hide:
		if (moves(_cards) or moves(_clearings)) {
			update();
		}
		break;
	default:
		break;
	}
	return QWidget::eventFilter(watched, event);
}
