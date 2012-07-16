
#include <iostream>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>

#include <tbb/tick_count.h>

#include "massive_common.h"


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


	PVHive::PVHive &hive = PVHive::PVHive::get();


	Block_p *blocks = new Block_p [obj_num];
	for (int i = 0; i < obj_num; ++i) {
		blocks[i] = Block_p(new Block(prop_num));
	}


	std::cout << "# registering objects" << std::endl;
	t1 = tbb::tick_count::now();
	for (int i = 0; i < obj_num; ++i) {
		hive.register_object(blocks[i]);
	}
	t2 = tbb::tick_count::now();
	print_stat("objects registered", t1, t2, obj_num);


	std::cout << "# creating properties" << std::endl;
	t1 = tbb::tick_count::now();
	for (int j = 0; j < obj_num; ++j) {
		for (int i = 0; i < prop_num; ++i) {
			hive.register_object(blocks[j],
			                     std::bind(&get_prop, std::placeholders::_1, i));
		}
	}
	t2 = tbb::tick_count::now();
	print_stat("properties registered", t1, t2, obj_prop_num);


	BlockAct *block_actors = new BlockAct [obj_act_num];
	std::cout << "# creating object actors" << std::endl;
	index = 0;
	t1 = tbb::tick_count::now();
	for (int j = 0; j < obj_num; ++j) {
		for (int i = 0; i < act_per_obj; ++i) {
			hive.register_actor(blocks[j], block_actors[index]);
			++index;
		}
	}
	t2 = tbb::tick_count::now();
	print_stat("object actors registered", t1, t2, obj_act_num);


	PropertyAct *prop_actors = new PropertyAct [prop_act_num];
	index = 0;
	for (int k = 0; k < obj_num; ++k) {
		for (int j = 0; j < prop_num; ++j) {
			for (int i = 0; i < act_per_prop; ++i) {
				prop_actors[index] = PropertyAct(rand() % prop_num);
				++index;
			}
		}
	}

	std::cout << "# creating property actors" << std::endl;
	index = 0;
	t1 = tbb::tick_count::now();
	for (int k = 0; k < obj_num; ++k) {
		for (int j = 0; j < prop_num; ++j) {
			for (int i = 0; i < act_per_prop; ++i) {
				hive.register_actor(blocks[k],
				                    prop_actors[index]);
				++index;
			}
		}
	}
	t2 = tbb::tick_count::now();
	print_stat("object actors registered", t1, t2, prop_act_num);



	BlockObs *block_observers = new BlockObs [obj_obs_num];
	std::cout << "# creating object observers" << std::endl;
	index = 0;
	t1 = tbb::tick_count::now();
	for (int j = 0; j < obj_num; ++j) {
		for (int i = 0; i < obs_per_obj; ++i) {
			hive.register_observer(blocks[j], block_observers[index]);
			++index;
		}
	}
	t2 = tbb::tick_count::now();
	print_stat("object observers registered", t1, t2, obj_obs_num);



	PropertyObs *prop_observers = new PropertyObs [prop_obs_num];

	std::cout << "# creating property observers" << std::endl;
	index = 0;
	t1 = tbb::tick_count::now();
	for (int k = 0; k < obj_num; ++k) {
		for (int j = 0; j < prop_num; ++j) {
			for (int i = 0; i < obs_per_prop; ++i) {
				hive.register_observer(blocks[k], std::bind(&get_prop, std::placeholders::_1, j),
				                       prop_observers[index]);
				++index;
			}
	}
	}
	t2 = tbb::tick_count::now();
	print_stat("property observers registered", t1, t2, prop_obs_num);


	return 0;
}
