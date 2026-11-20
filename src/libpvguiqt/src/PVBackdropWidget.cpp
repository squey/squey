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
#include <QPainter>
#include <QPolygonF>

#include <algorithm>
#include <array>
#include <random>

namespace
{

/** As the theme rounds the corners of cards, in pixels. */
constexpr int CARD_RADIUS = 8;
/** Between the axes of the drawing, in pixels. */
constexpr int AXIS_SPACING = 160;

/** Rows made up for the drawing: groups of close rows, one of them selected. */
struct Rows {
	static constexpr int AXES = 32;

	struct Row {
		std::array<float, AXES> values;
		bool selected;
	};

	Rows()
	{
		// The same drawing in every page.
		std::mt19937 random(5);
		std::uniform_real_distribution<float> uniform(0.08f, 0.92f);
		std::normal_distribution<float> step(0.f, 0.16f);
		constexpr int GROUPS = 6;
		for (int group = 0; group < GROUPS; group++) {
			std::array<float, AXES> profile;
			profile[0] = uniform(random);
			for (int axis = 1; axis < AXES; axis++) {
				profile[axis] = std::clamp(profile[axis - 1] + step(random), 0.05f, 0.95f);
			}
			std::normal_distribution<float> spread(0.f, 0.015f + 0.008f * float(group));
			for (int row = 0; row < 12 + 4 * group; row++) {
				Row& added = rows.emplace_back(Row{.values = {}, .selected = group == 2});
				for (int axis = 0; axis < AXES; axis++) {
					added.values[axis] = std::clamp(profile[axis] + spread(random), 0.f, 1.f);
				}
			}
		}
		for (int row = 0; row < 12; row++) {
			Row& outlier = rows.emplace_back(Row{.values = {}, .selected = false});
			for (float& value : outlier.values) {
				value = uniform(random);
			}
		}
	}

	std::vector<Row> rows;
};

/**
 * The drawing of a widget of @p size, for a screen of @p ratio: axes and rows in @p color,
 * selected rows in @p selection_color.
 */
QPixmap decoration(QSize size, qreal ratio, QColor const& color, QColor const& selection_color)
{
	static const Rows made_up;
	QPixmap pixmap(size * ratio);
	pixmap.setDevicePixelRatio(ratio);
	pixmap.fill(Qt::transparent);
	QPainter painter(&pixmap);
	painter.setRenderHint(QPainter::Antialiasing);

	// Axes from one edge to the other, as many on each side of the center.
	const int axes = std::min(Rows::AXES, size.width() / AXIS_SPACING + 2);
	const qreal first = (size.width() - (axes - 1) * AXIS_SPACING) / 2.;
	const qreal top = size.height() * 0.06;
	const qreal height = size.height() * 0.88;
	const auto x_of = [first](int axis) { return first + axis * AXIS_SPACING; };

	QColor axis_color = color;
	axis_color.setAlphaF(0.12f);
	painter.setPen(QPen(axis_color, 1));
	for (int axis = 0; axis < axes; axis++) {
		painter.drawLine(QPointF(x_of(axis), top), QPointF(x_of(axis), top + height));
	}
	// The selection over the other rows, as Squey draws it.
	for (const bool selected : {false, true}) {
		QColor row_color = selected ? selection_color : color;
		row_color.setAlphaF(selected ? 0.28f : 0.045f);
		painter.setPen(QPen(row_color, 1.2));
		for (const Rows::Row& row : made_up.rows) {
			if (row.selected != selected) {
				continue;
			}
			QPolygonF line;
			for (int axis = 0; axis < axes; axis++) {
				line << QPointF(x_of(axis), top + height * (1.f - row.values[axis]));
			}
			painter.drawPolyline(line);
		}
	}
	return pixmap;
}

} // namespace

PVGuiQt::PVBackdropWidget::PVBackdropWidget(QWidget* parent)
    : QWidget(parent), _decoration_color(palette().color(QPalette::Highlight))
{
}

void PVGuiQt::PVBackdropWidget::set_decoration_color(QColor const& color)
{
	_decoration_color = color;
	_decoration = {};
	update();
}

void PVGuiQt::PVBackdropWidget::add_card(QWidget* card)
{
	_cards.emplace_back(card);
	for (QWidget* moving = card; moving != nullptr and moving != this;
	     moving = moving->parentWidget()) {
		moving->installEventFilter(this);
	}
}

void PVGuiQt::PVBackdropWidget::paintEvent(QPaintEvent*)
{
	QPainter painter(this);
	painter.fillRect(rect(), palette().color(QPalette::Window));
	const qreal ratio = devicePixelRatioF();
	if (_decoration.size() != size() * ratio) {
		_decoration =
		    decoration(size(), ratio, palette().color(QPalette::WindowText), _decoration_color);
	}
	painter.drawPixmap(0, 0, _decoration);

	// A shadow, wider and fainter from each card, a bit below it.
	painter.setRenderHint(QPainter::Antialiasing);
	painter.setPen(Qt::NoPen);
	painter.setBrush(QColor(0, 0, 0, 7));
	for (const QPointer<QWidget>& card : _cards) {
		if (card == nullptr or not card->isVisible()) {
			continue;
		}
		const QRectF area(card->mapTo(this, QPoint(0, 0)), card->size());
		for (int spread = 1; spread <= 8; spread++) {
			painter.drawRoundedRect(area.adjusted(-spread, 2 - spread, spread, 2 + spread),
			                        CARD_RADIUS + spread, CARD_RADIUS + spread);
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

bool PVGuiQt::PVBackdropWidget::eventFilter(QObject* watched, QEvent* event)
{
	const auto moves_card = [this, watched]() {
		return std::any_of(_cards.begin(), _cards.end(), [watched](QPointer<QWidget> const& card) {
			return card == watched or (card != nullptr and watched->isWidgetType() and
			                           static_cast<QWidget*>(watched)->isAncestorOf(card));
		});
	};
	switch (event->type()) {
	case QEvent::Move:
	case QEvent::Resize:
	case QEvent::Show:
	case QEvent::Hide:
		if (moves_card()) {
			update();
		}
		break;
	default:
		break;
	}
	return QWidget::eventFilter(watched, event);
}
