/* * MIT License
 *
 * © Squey, 2026
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

#ifndef __PVGUIQT_PVBACKDROPWIDGET_H__
#define __PVGUIQT_PVBACKDROPWIDGET_H__

#include <pvguiqt/export.h>

#include <QColor>
#include <QPixmap>
#include <QPointer>
#include <QWidget>

#include <vector>

namespace PVGuiQt
{

/**
 * A widget that draws, behind what it shows, parallel coordinates of made-up rows, the plot Squey
 * is known for, in the colors of the theme: those of its background and of its text, and
 * decoration_color for the rows of a selection. What stands out of it, as cards, gets a shadow.
 */
class PVGUIQT_EXPORT PVBackdropWidget : public QWidget
{
	Q_OBJECT
	Q_PROPERTY(QColor decoration_color READ decoration_color WRITE set_decoration_color)

  public:
	explicit PVBackdropWidget(QWidget* parent = nullptr);

  public:
	QColor decoration_color() const { return _decoration_color; }
	void set_decoration_color(QColor const& color);

  protected:
	/**
	 * Gives a shadow to @p card, a widget in this one with the corners the theme rounds cards
	 * with, while it shows. Called once @p card is laid out where it stays: the shadow follows it
	 * as it moves with what holds it then.
	 */
	void add_card(QWidget* card);
	/**
	 * Clears the drawing around @p widget, a widget in this one that shows on the background
	 * alone: the drawing fades out a bit beyond its edges, out of the room the layout gives it.
	 * Follows @p widget as add_card() does.
	 */
	void add_clearing(QWidget* widget);

	void paintEvent(QPaintEvent* event) override;
	void changeEvent(QEvent* event) override;
	bool eventFilter(QObject* watched, QEvent* event) override;

  private:
	/** Paints the widget again as @p widget, or what holds it, moves or shows. */
	void follow(QWidget* widget);

  private:
	QColor _decoration_color;
	/** The drawing, for the size and the colors of the widget it was drawn for. */
	QPixmap _decoration;
	std::vector<QPointer<QWidget>> _cards;
	std::vector<QPointer<QWidget>> _clearings;
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVBACKDROPWIDGET_H__
