/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef _PVCORE_LAMBDA_CONNECT_H__
#define _PVCORE_LAMBDA_CONNECT_H__

#include <functional>
#include <QObject>

class connect_functor_helper : public QObject
{
	Q_OBJECT

  public:
	connect_functor_helper(QObject* parent, const std::function<void()>& f)
	    : QObject(parent), _function(f)
	{
	}

  public Q_SLOTS:
	void signaled() { _function(); }

  private:
	std::function<void()> _function;
};

template <class T>
bool connect(QObject* sender,
             const char* signal,
             const T& receiver,
             Qt::ConnectionType type = Qt::AutoConnection)
{
	return QObject::connect(sender, signal, new connect_functor_helper(sender, receiver),
	                        SLOT(signaled()), type);
}

#endif /* _PVCORE_LAMBDA_CONNECT_H__ */
