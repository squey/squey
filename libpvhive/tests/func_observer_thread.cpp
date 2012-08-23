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

void boost_thread(MyClass::shared_pointer test_sp)
{
	// TODO: Exit the thread properly on dialog close.
	uint32_t counter;
	do {
		counter = test_sp->get_counter();
		PVHive::call<FUNC(MyClass::set_counter)>(test_sp, counter+1);
		sleep(1);
	} while (counter < 99);
}

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	MyClass::shared_pointer test_sp(new MyClass);

	TestDlg test_dlg(nullptr, test_sp);
	test_dlg.show();

	boost::thread th(boost::bind(boost_thread,  boost::ref(test_sp)));

	app.exec();

	return 0;
}

void set_counter_Observer::update(arguments_deep_copy_type const& args) const
{
	uint32_t counter = std::get<0>(args);

	if (_parent->thread() == QThread::currentThread()) {
		std::cout << "set_counter_Observer::update = " << counter << " QT THREAD :-)" << std::endl;
		_parent->update_counter(counter);
	}
	else {
		std::cout << "set_counter_Observer::update = " << counter << " BOOST THREAD :-(" << std::endl;
	}
}
