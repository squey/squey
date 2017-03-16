/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2017
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
