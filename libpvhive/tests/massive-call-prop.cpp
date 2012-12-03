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

#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_stat.h>
#include <pvkernel/core/picviz_assert.h>

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
	char title[512];

	long obs_count = prop_num * obs_num;

	Block_p block = Block_p(new Block(prop_num));
	PropertyObs *observers = new PropertyObs [obs_count];
	PropertyAct *actors = new PropertyAct [prop_num];

	PVHive::PVHive &hive = PVHive::PVHive::get();


	std::cout << "# init" << std::endl;
	hive.register_object(block);

	for (int i = 0; i < prop_num; ++i) {
		hive.register_object(block, std::bind(&get_prop, std::placeholders::_1, i));
	}

	for (int i = 0; i < prop_num; ++i) {
		for (int j = 0; j < obs_num; ++j) {
			hive.register_observer(block, std::bind(&get_prop, std::placeholders::_1, i),
			                       observers[obs_num * i + j]);
		}
	}

	for (long i = 0; i < prop_num; ++i) {
		actors[i] = PropertyAct(i % prop_num, 42);
		hive.register_actor(block, actors[i]);
	}


	long v = 100, ov = -1;
	std::cout << "# doing calls (it can take a while)" << std::endl;
	BENCH_START(calls);
	for (long n = 0; n < action_num; ++n) {
		for (int i = 0; i < prop_num; ++i) {
			actors[i].action();
		}
		v = (100 * n) / action_num;
		if (v != ov) {
			ov = v;
		}
	}
	BENCH_STOP(calls);

	if (obs_num != 0) {
		snprintf(title, 512, "refreshments_%d_%d_%d", action_num, prop_num, obs_num);
		PV_STAT_CALLS(title, (long)action_num * obs_count / BENCH_END_TIME(calls));
	} else {
		snprintf(title, 512, "calls_%d_%d_%d", action_num, prop_num, obs_num);
		PV_STAT_CALLS(title, (long)action_num * prop_num / BENCH_END_TIME(calls));
	}

	std::cout << "# validating" << std::endl;
	int val = actors[0].get_value();
	for (long i = 0; i < obs_count; ++i) {
		int vv = observers[i].get_value();
		PV_VALID(vv, val, "i", i);
	}

	return 0;
}


