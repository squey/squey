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

#ifndef __PVWIDGETS_PVUTILS_H__
#define __PVWIDGETS_PVUTILS_H__

#include <QFont>
#include <QString>

#define SQUEY_TOOLTIP_MAX_WIDTH 800

namespace PVWidgets
{

namespace PVUtils
{
QString shorten_path(const QString& s, const QFont& font, uint64_t nb_px);
void html_word_wrap_text(QString& string, const QFont& font, uint64_t nb_px);

/*! \brief Returns the maximum tooltip width related to a widget
 */
uint32_t tooltip_max_width(QWidget* w);

QString bytes_to_human_readable(size_t byte_count);

} // namespace PVUtils
;
} // namespace PVWidgets

#endif // __PVWIDGETS_PVUTILS_H__
