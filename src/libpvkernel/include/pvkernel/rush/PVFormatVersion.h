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

#ifndef PVFORMAT_VERSION_H
#define PVFORMAT_VERSION_H

#include <vector>
#include <QDomDocument>
#include <QString>

namespace PVRush
{

namespace PVFormatVersion
{
void to_current(QDomDocument& doc);

namespace __impl
{
void from0to1(QDomDocument& doc);
void from1to2(QDomDocument& doc);
void from2to3(QDomDocument& doc);
void from3to4(QDomDocument& doc);
void from4to5(QDomDocument& doc);
void from5to6(QDomDocument& doc);
void from6to7(QDomDocument& doc);
void from7to8(QDomDocument& doc);
void from8to9(QDomDocument& doc);
void from9to10(QDomDocument& doc);
void from10to11(QDomDocument& doc);

void _rec_0to1(QDomElement doc);
void _rec_1to2(QDomElement doc);
void _rec_2to3(QDomElement doc);
void _rec_3to4(QDomNode doc);
void _rec_4to5(QDomNode doc);
void _rec_5to6(QDomNode doc);
QString get_version(QDomDocument const& doc);
} // namespace __impl
} // namespace PVFormatVersion
;
} // namespace PVRush

#endif
