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

#ifndef PVCORE_QMETAOBJECTHELPER_H
#define PVCORE_QMETAOBJECTHELPER_H

#include <QObject>

namespace PVCore
{

/**
 * These template functions aims to overcome the missing of compile-time checking version of
 * QMetaObject::invokeMethod(...)
 */
template <typename R, typename F, typename... A>
void invokeMethod(R* r, const F& f, Qt::ConnectionType type = Qt::AutoConnection, A&&... a)
{
	QObject src;
	QObject::connect(&src, &QObject::destroyed, r, [=]() { (r->*f)(a...); }, type);
}
}

#endif // PVCORE_QMETAOBJECTHELPER_H
