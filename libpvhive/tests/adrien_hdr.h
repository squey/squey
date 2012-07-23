/**
 * \file adrien_hdr.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef TEST_ADRIEN_HDR_H
#define TEST_ADRIEN_HDR_H

#include <QObject>
#include <QThread>
#include <QTimer>

#include <iostream>

#include <boost/thread.hpp>

#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>

#include "adrien_objs.h"

class MyObjObserver: public PVHive::PVObserver<MyObject>
{
public:
	void refresh() { std::cout << "  MyObjObserver refresh to i=" << get_object()->get_i() << std::endl; }
	void about_to_be_deleted() { }
};

class MyObjActor: public PVHive::PVActor<MyObject>
{
public:
#if 0
	template <typename F, F f, typename... Ttypes>
	void call(Ttypes... params)
	{
		Actor<MyObject>::call<F, f>(params...);
		/*if (f == &MyObject::set_i2) {
			std::cout << "actor int2 custom" << std::endl;
		}
		else
		if (f == &MyObject::set_prop) {
			parent_cc().refresh_observers(&_p._prop);
		}*/
	}
#endif

	/*
	template <typename... Ttypes>
	void call(decltype(&MyObject::set_i2) f, Ttypes... params)
	{
		std::cout << "set_i2 special actor" << std::endl;
		Actor<MyObject>::call(f, params...);
	}

	template <typename... Ttypes>
	void call(decltype(&MyObject::set_i) f, Ttypes... params)
	{
		std::cout << "set_i special actor" << std::endl;
		Actor<MyObject>::call(f, params...);
	}*/
};

class MyThread : public QThread
{
	Q_OBJECT

public:
	MyThread(MyObject_p &o, QObject *parent = 0) :
		QThread(parent),
		_o(o),
		_c(0)
	{
		PVHive::PVHive::get().register_actor(o, _actor);
		_timer = new QTimer(this);
		connect(_timer, SIGNAL(timeout()), this, SLOT(update_prop()));
		_timer->start(1000);
	}

	~MyThread()
	{
		quit();
		wait();
	}

	void run()
	{
		std::cout << "Update thread is " << boost::this_thread::get_id() << std::endl;
		exec();
	}

public slots:
	void update_prop()
	{
		std::cout << "Update prop to " << _c << std::endl;
		_actor.call<decltype(&MyObject::set_prop), &MyObject::set_prop>(boost::cref(ObjectProperty(_c)));
		++_c;
	}

private:
	MyObject_p _o;
	int _c;
	MyObjActor _actor;
	QTimer *_timer;
};

#endif // TEST_ADRIEN_HDR_H
