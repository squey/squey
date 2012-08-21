/**
 * \file func_observer_thread.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <boost/thread.hpp>

#include <QApplication>
#include <QThread>

#include <pvhive/PVCallHelper.h>
#include <pvkernel/core/PVSharedPointer.h>

#include "func_observer_thread.h"

void boost_thread(Test::shared_pointer test_sp)
{
	uint32_t counter;
	do {
		counter = test_sp->get_counter();
		PVHive::call<FUNC(Test::set_counter)>(test_sp, counter+1);
		sleep(1);
	} while (counter < 99);
}

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	Test::shared_pointer test_sp(new Test);

	TestDlg test_dlg(nullptr, test_sp);
	test_dlg.show();

	boost::thread th(boost::bind(boost_thread,  boost::ref(test_sp)));

	app.exec();

	return 0;
}

void set_counter_Observer::update(arguments_type const& args) const
{
	uint32_t counter = args.get_arg<0>();

	if (_parent->thread() == QThread::currentThread()) {
		std::cout << "set_counter_Observer::update = " << counter << " QT THREAD :-)" << std::endl;
		_parent->update_counter(counter);
	}
	else {
		std::cout << "set_counter_Observer::update = " << counter << " BOOST THREAD :-(" << std::endl;
	}
}
