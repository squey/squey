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

#ifndef PCAP_GUI_STYLE_H
#define PCAP_GUI_STYLE_H

#include <QColor>
#include <QString>

#include <cstdint>
#include <string_view>

class QLabel;
class QPainter;
class QPushButton;
class QRect;
class QWidget;

/**
 * What the parts of the pcap import dialog share of their look, which follows the light or
 * dark theme of Squey.
 */
namespace PVPcapsicum::style
{

//! Of text saying less than the text around it.
QColor secondary();
//! That of links.
QColor accent();
QColor error();
QColor warning();
QColor success();

//! A link of rich text, drawn as the about box of Squey draws its links.
QString link(const QString& target, const QString& text);

//! Draws the label as Squey draws the button to press once all is set.
void make_primary(QPushButton* button);

//! Draws the text of the label in the color, and in a smaller font if `small`.
void color_label(QLabel* label, const QColor& color, bool small = false);

/**
 * Draws a small rounded tag holding `text`, such as "built-in", its left at `left`, centered
 * vertically in `line`.
 *
 * @return its width
 */
int paint_tag(
    QPainter& painter, int left, const QRect& line, const QString& text, const QColor& color);

//! Such as "54.2 MB".
QString size_text(uint64_t bytes);
//! Such as "1,234".
QString count_text(uint64_t count);
//! Such as "1.2 M", for counts that are estimates.
QString short_count_text(uint64_t count);
//! Such as "2008-07-22 01:51", in UTC.
QString time_text(int64_t nanoseconds);

//! What Squey makes of a type of tshark, in a word: time, number, IPv4, text...
QString type_text(std::string_view tshark_type);

QString to_qstring(std::string_view text);

} // namespace PVPcapsicum::style

#endif // PCAP_GUI_STYLE_H
