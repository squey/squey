/**
 * \file massive-register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>

#include <tbb/tick_count.h>

#include "massive_common.h"

#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_stat.h>

/*****************************************************************************
 * main
 *****************************************************************************/

int main(int argc, char **argv)
{
	tbb::tick_count t1, t2;
	long index;

	if (argc <= 6) {
		std::cerr << "usage: " << argv[0] << " objects_number properties_number actor_per_object actor_per_property observers_per_object observers_per_property" << std::endl;
		return 1;
	}

	int obj_num = atoi(argv[1]);
	int prop_num = atoi(argv[2]);

	int act_per_obj = atoi(argv[3]);
	int act_per_prop = atoi(argv[4]);

	int obs_per_obj = atoi(argv[5]);
	int obs_per_prop = atoi(argv[6]);

	long obj_prop_num = obj_num * prop_num;

	long obj_act_num = obj_num * act_per_obj;
	long obj_obs_num = obj_num * obs_per_obj;

	long prop_act_num = obj_prop_num * act_per_prop;
	long prop_obs_num = obj_prop_num * obs_per_prop;

	if(obj_num == 0)  {
		std::cerr << "at least one object must be created" << std::endl;
		return 1;
	}

	PVHive::PVHive &hive = PVHive::PVHive::get();


	Block_p *blocks = new Block_p [obj_num];
	for (int i = 0; i < obj_num; ++i) {
		blocks[i] = Block_p(new Block(prop_num));
	}


	if(((prop_num == 0)
	    && (act_per_obj == 0) && (act_per_prop == 0)
	    && (obs_per_obj == 0) && (obs_per_prop == 0))) {
		std::stringstream ss;
		ss << "register_object_" << obj_num;
		std::cout << "# registering objects" << std::endl;
		BENCH_START(calls);
		t1 = tbb::tick_count::now();
		for (int i = 0; i < obj_num; ++i) {
			hive.register_object(blocks[i]);
		}
		BENCH_STOP(calls);
		PV_STAT_CALLS(ss.str(), obj_num / BENCH_END_TIME(calls));
	}

	if(((prop_num != 0)
	    && (act_per_obj == 0) && (act_per_prop == 0)
	    && (obs_per_obj == 0) && (obs_per_prop == 0))) {
		std::stringstream ss;
		ss << "register_property_" << obj_num << "_" << prop_num;
		std::cout << "# creating properties" << std::endl;
		BENCH_START(calls);
		for (int j = 0; j < obj_num; ++j) {
			for (int i = 0; i < prop_num; ++i) {
				hive.register_object(blocks[j],
				                     std::bind(&get_prop, std::placeholders::_1, i));
			}
		}
		BENCH_STOP(calls);
		PV_STAT_CALLS(ss.str(), obj_prop_num / BENCH_END_TIME(calls));
	}


	BlockAct *block_actors = new BlockAct [obj_act_num];
	if(((prop_num == 0)
	    && (act_per_obj != 0) && (act_per_prop == 0)
	    && (obs_per_obj == 0) && (obs_per_prop == 0))) {
		std::stringstream ss;
		ss << "register_object_actor_" << obj_num << "_" << act_per_obj;
		std::cout << "# creating object actors" << std::endl;
		index = 0;
		BENCH_START(calls);
		for (int j = 0; j < obj_num; ++j) {
			for (int i = 0; i < act_per_obj; ++i) {
				hive.register_actor(blocks[j], block_actors[index]);
				++index;
			}
		}
		BENCH_STOP(calls);
		PV_STAT_CALLS(ss.str(), obj_act_num / BENCH_END_TIME(calls));
	}


	PropertyAct *prop_actors = new PropertyAct [prop_act_num];
	index = 0;
	for (int k = 0; k < obj_num; ++k) {
		for (int j = 0; j < prop_num; ++j) {
			for (int i = 0; i < act_per_prop; ++i) {
				prop_actors[index] = PropertyAct(rand() % prop_num, 42);
				++index;
			}
		}
	}

	if(((prop_num != 0)
	    && (act_per_obj == 0) && (act_per_prop != 0)
	    && (obs_per_obj == 0) && (obs_per_prop == 0))) {
		std::stringstream ss;
		ss << "register_property_actor_" << obj_num << "_" << prop_num << "_" << act_per_prop;
		std::cout << "# creating property actors" << std::endl;
		index = 0;
		BENCH_START(calls);
		for (int k = 0; k < obj_num; ++k) {
			for (int j = 0; j < prop_num; ++j) {
				for (int i = 0; i < act_per_prop; ++i) {
					hive.register_actor(blocks[k],
					                    prop_actors[index]);
					++index;
				}
			}
		}
		BENCH_STOP(calls);
		PV_STAT_CALLS(ss.str(), prop_act_num / BENCH_END_TIME(calls));
	}


	BlockObs *block_observers = new BlockObs [obj_obs_num];
	if(((prop_num == 0)
	    && (act_per_obj == 0) && (act_per_prop == 0)
	    && (obs_per_obj != 0) && (obs_per_prop == 0))) {
		std::stringstream ss;
		ss << "register_object_observer_" << obj_num << "_" << obs_per_obj;
		std::cout << "# creating object observers" << std::endl;
		index = 0;
		BENCH_START(calls);
		for (int j = 0; j < obj_num; ++j) {
			for (int i = 0; i < obs_per_obj; ++i) {
				hive.register_observer(blocks[j], block_observers[index]);
				++index;
			}
		}
		BENCH_STOP(calls);
		PV_STAT_CALLS(ss.str(), obj_obs_num / BENCH_END_TIME(calls));
	}


	PropertyObs *prop_observers = new PropertyObs [prop_obs_num];

	if(((prop_num != 0)
	    && (act_per_obj == 0) && (act_per_prop == 0)
	    && (obs_per_obj == 0) && (obs_per_prop != 0))) {
		std::stringstream ss;
		ss << "register_property_observer_" << obj_num << "_" << prop_num << "_" << obs_per_prop;
		std::cout << "# creating property observers" << std::endl;
		index = 0;
		BENCH_START(calls);
		for (int k = 0; k < obj_num; ++k) {
			for (int j = 0; j < prop_num; ++j) {
				for (int i = 0; i < obs_per_prop; ++i) {
					hive.register_observer(blocks[k], std::bind(&get_prop,
					                                            std::placeholders::_1,
					                                            j),
					                       prop_observers[index]);
					++index;
				}
			}
		}
		BENCH_STOP(calls);
		PV_STAT_CALLS(ss.str(), prop_obs_num / BENCH_END_TIME(calls));
	}

	return 0;
}
