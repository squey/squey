
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
		std::cerr << "usage: " << argv[0] << " actions_per_actors actors_number observers" << std::endl;
		return 1;
	}

	int action_num = atoi(argv[1]);
	int actor_num = atoi(argv[2]);
	int obs_num = atoi(argv[3]);

	Block_p block = Block_p(new Block(1));
	BlockObs *observers = new BlockObs [obs_num];
	BlockAct *actors = new BlockAct [actor_num];

	PVHive::PVHive &hive = PVHive::PVHive::get();


	std::cout << "# init" << std::endl;
	std::cout << "#   registering objects" << std::endl;
	hive.register_object(block);


	std::cout << "#   registering observers" << std::endl;
	for (int j = 0; j < obs_num; ++j) {
		hive.register_observer(block, observers[j]);
	}

	std::cout << "#   registering actors" << std::endl;
	for (long i = 0; i < actor_num; ++i) {
		actors[i] = BlockAct();
		hive.register_actor(block, actors[i]);
	}


	long v = 100, ov = -1;
	std::cout << "# doing calls" << std::endl;
	t1 = tbb::tick_count::now();
	for (long n = 0; n < action_num; ++n) {
		for (int i = 0; i < actor_num; ++i) {
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
		print_stat("refreshments", t1, t2, (long)action_num * actor_num * obs_num);
	} else {
		print_stat("calls", t1, t2, (long)action_num * actor_num);
	}

	std::cout << "# checking" << std::endl;
	int val = actors[0].get_value();
	for (long i = 0; i < obs_num; ++i) {
		int vv = observers[i].get_value();
		if(vv != val) {
			std::cout << "  error with observer " << i
			          << " (" << vv << ")" << std::endl;
		}
	}

	return 0;
}


