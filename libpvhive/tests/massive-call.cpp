
#include <iostream>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>

#include "massive_common.h"

/*****************************************************************************
 * main
 *****************************************************************************/

int main(int argc, char **argv)
{
	tbb::tick_count t1, t2;

	if (argc <= 4) {
		std::cerr << "usage: " << argv[0] << " actions_per_actors actors_number properties_number observer_per_properties" << std::endl;
		return 1;
	}

	int action_num = atoi(argv[1]);
	int actor_num = atoi(argv[2]);
	int prop_num = atoi(argv[3]);
	int obs_num = atoi(argv[4]);

	long obs_count = prop_num * obs_num;

	Block_p block = Block_p(new Block(prop_num));
	PropertyObs *observers = new PropertyObs [obs_count];
	PropertyAct *actors = new PropertyAct [actor_num];

	PVHive::PVHive &hive = PVHive::PVHive::get();


	std::cout << "# init" << std::endl;
	std::cout << "#   creating objects" << std::endl;
	hive.register_object(block);
	for (int i = 0; i < prop_num; ++i) {
		hive.register_object(block, std::bind(&get_prop, std::placeholders::_1, i));
	}

	std::cout << "#   creating properties" << std::endl;
	hive.register_object(block);
	for (int i = 0; i < prop_num; ++i) {
		hive.register_object(block, std::bind(&get_prop, std::placeholders::_1, i));
	}


	std::cout << "#   creating observers" << std::endl;
	for (int i = 0; i < prop_num; ++i) {
		for (int j = 0; j < obs_num; ++j) {
			hive.register_observer(block, std::bind(&get_prop, std::placeholders::_1, i),
			                       observers[obs_num * i + j]);
		}
	}


	std::cout << "#   creating actors" << std::endl;
	for (int i = 0; i < actor_num; ++i) {
		actors[i] = PropertyAct(rand() % prop_num);
		hive.register_actor(block, actors[i]);
	}


	int v, ov = -1;
	std::cout << "# doing calls" << std::endl;
	t1 = tbb::tick_count::now();
	for (int n = 0; n < action_num; ++n) {
		for (int i = 0; i < actor_num; ++i) {
			actors[i].action();
		}
		v = (100 * n) / action_num;
		if (v != ov) {
			printf("\rprogress: %4d %%", v);
			fflush(stdout);
			ov = v;
		}
	}
	std::cout << std::endl;

	t2 = tbb::tick_count::now();
	print_stat("refresh", t1, t2, (long)action_num * actor_num * obs_num);

	return 0;
}
