/**
 * \file massive-call-prop.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

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

	if (argc <= 3) {
		std::cerr << "usage: " << argv[0] << " actions_per_property properties_number observer_per_properties" << std::endl;
		return 1;
	}

	int action_num = atoi(argv[1]);
	int prop_num = atoi(argv[2]);
	int obs_num = atoi(argv[3]);

	long obs_count = prop_num * obs_num;

	Block_p block = Block_p(new Block(prop_num));
	PropertyObs *observers = new PropertyObs [obs_count];
	PropertyAct *actors = new PropertyAct [prop_num];

	PVHive::PVHive &hive = PVHive::PVHive::get();


	std::cout << "# init" << std::endl;
	std::cout << "#   registering objects" << std::endl;
	hive.register_object(block);


	std::cout << "#   registering properties" << std::endl;
	for (int i = 0; i < prop_num; ++i) {
		hive.register_object(block, std::bind(&get_prop, std::placeholders::_1, i));
	}


	std::cout << "#   registering observers" << std::endl;
	for (int i = 0; i < prop_num; ++i) {
		for (int j = 0; j < obs_num; ++j) {
			hive.register_observer(block, std::bind(&get_prop, std::placeholders::_1, i),
			                       observers[obs_num * i + j]);
		}
	}

	std::cout << "#   registering actors" << std::endl;
	for (long i = 0; i < prop_num; ++i) {
		actors[i] = PropertyAct(i % prop_num, 42);
		hive.register_actor(block, actors[i]);
	}


	long v = 100, ov = -1;
	std::cout << "# doing calls" << std::endl;
	t1 = tbb::tick_count::now();
	for (long n = 0; n < action_num; ++n) {
		for (int i = 0; i < prop_num; ++i) {
			actors[i].action();
		}
		v = (100 * n) / action_num;
		if (v != ov) {
			printf("\rprogress: %4ld %%", v);
			fflush(stdout);
			ov = v;
		}
	}
	printf("\rprogress: %4ld %%\n", v);

	t2 = tbb::tick_count::now();
	if (obs_num != 0) {
		print_stat("refreshments", t1, t2, (long)action_num * obs_count);
	} else {
		print_stat("calls", t1, t2, (long)action_num * prop_num);
	}

	std::cout << "# checking" << std::endl;
	int val = actors[0].get_value();
	for (long i = 0; i < obs_count; ++i) {
		int vv = observers[i].get_value();
		if(vv != val) {
			std::cout << "  error with observer " << i
			          << " (" << vv << ")" << std::endl;
		}
	}

	return 0;
}


