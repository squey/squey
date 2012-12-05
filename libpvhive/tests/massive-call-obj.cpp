/**
 * \file massive-call-obj.cpp
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
		std::cerr << "usage: " << argv[0] << " actions_per_actors actors_number observers" << std::endl;
		return 1;
	}

	int action_num = atoi(argv[1]);
	int actor_num = atoi(argv[2]);
	int obs_num = atoi(argv[3]);
	char title[512];

	Block_p block = Block_p(new Block(1));
	BlockObs *observers = new BlockObs [obs_num];
	BlockAct *actors = new BlockAct [actor_num];

	PVHive::PVHive &hive = PVHive::PVHive::get();


	std::cout << "# init" << std::endl;
	hive.register_object(block);

	for (int j = 0; j < obs_num; ++j) {
		hive.register_observer(block, observers[j]);
	}

	for (long i = 0; i < actor_num; ++i) {
		actors[i] = BlockAct();
		hive.register_actor(block, actors[i]);
	}


	long v = 100, ov = -1;
	std::cout << "# doing calls (it can take a while)" << std::endl;
	BENCH_START(calls);
	for (long n = 0; n < action_num; ++n) {
		for (int i = 0; i < actor_num; ++i) {
			actors[i].action();
		}
		v = (100 * n) / action_num;
		if (v != ov) {
			ov = v;
		}
	}
	BENCH_STOP(calls);
	if (obs_num != 0) {
		snprintf(title, 512, "refreshments_%d_%d_%d", action_num, actor_num, obs_num);
		PV_STAT_CALLS(title, (long)action_num * actor_num / BENCH_END_TIME(calls));
	} else {
		snprintf(title, 512, "calls_%d_%d_%d", action_num, actor_num, obs_num);
		PV_STAT_CALLS(title, (long)action_num * actor_num / BENCH_END_TIME(calls));
	}

	std::cout << "# validating" << std::endl;
	int val = actors[0].get_value();
	for (long i = 0; i < obs_num; ++i) {
		int vv = observers[i].get_value();
		PV_VALID(vv, val, "i", i);
	}

	return 0;
}


