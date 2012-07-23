/**
 * \file data-in-thread.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>

#include <boost/thread.hpp>

#include <QApplication>

#include "data-in-thread_obj.h"
#include "data-in-thread_dlg.h"

Entity_p *static_e = nullptr;

typedef PVHive::PVActor<Entity> EntityActor;

void th_actor_func()
{
	std::cout << "th_actor: init - " << boost::this_thread::get_id()
	          << std::endl;
	int count = 0;
	Entity_p e = Entity_p(new Entity(42));

	static_e = &e;
	PVHive::PVHive::get().register_object(e);

	PVHive::PVHive::get().print();

	EntityActor a;
	PVHive::PVHive::get().register_actor(e, a);

	std::cout << "th_actor: pseudo sync" << std::endl;
	sleep(1);

	std::cout << "th_actor: run" << std::endl;
	while (count < 10) {
		sleep(1);
		std::cout << "th_actor_func - " << boost::this_thread::get_id()
		          << " - e.set_i(" << count << ")" << std::endl;
		PVACTOR_CALL(a, &Entity::set_i, count);
		++count;
	}

	std::cout << "th_actor: clean" << std::endl;
	PVHive::PVHive::get().unregister_actor(a);

	std::cout << "th_actor: terminate" << std::endl;
}

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	boost::thread tha(boost::bind(th_actor_func));
	sleep(1);

	TestDlg dlg(nullptr);

	dlg.show();

	app.exec();

	return 0;
}
