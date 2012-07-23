/**
 * \file modal.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserverSignal.h>

#include <QApplication>
#include <QDialog>

#include <boost/thread.hpp>

#include "modal_obj.h"
#include "modal_dlg.h"

Entity_p *shared_e = nullptr;

/*****************************************************************************
 * about the actor (not the artist)
 *****************************************************************************/

typedef PVHive::PVActor<Entity> EntityActor;

void th_actor_func()
{
	std::cout << "# actor: init - " << boost::this_thread::get_id()
	          << std::endl;
	int count = 0;
	Entity_p e = Entity_p(new Entity(42));

	shared_e = &e;
	PVHive::PVHive::get().register_object(e);

	EntityActor a;
	PVHive::PVHive::get().register_actor(e, a);

	std::cout << "# actor: pseudo sync" << std::endl;
	sleep(1);

	std::cout << "# actor: run" << std::endl;
	while (count < 10) {
		sleep(1);
		std::cout << "# actor_func - " << boost::this_thread::get_id()
		          << " - e.set_i(" << count << ")" << std::endl;
		PVACTOR_CALL(a, &Entity::set_i, count);
		++count;
	}

	std::cout << "# actor: clean" << std::endl;
	PVHive::PVHive::get().unregister_actor(a);

	std::cout << "# actor: terminate" << std::endl;
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
