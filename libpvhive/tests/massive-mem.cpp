
#include <iostream>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>

#include "massive_common.h"

void print_size(const char *text, size_t size)
{
	static const char *units[] = {"o", "Kio", "Mio", "Gio", "Tio", "Pio"};
	int index = 0;
	uint64_t scale = 1;
	size_t s = size;

	while (s > 1024) {
		scale *= 1024;
		++index;
		s /= 1024;
	}

	printf("%s: %3g %s\n", text, (double)size / scale, units[index]);
}

/*****************************************************************************
 * main
 *****************************************************************************/

#define OBJ_NUM 2048
#define PROP_NUM 2048

int main()
{
	size_t base, next;

	PVHive::PVHive &hive = PVHive::PVHive::get();
	base = hive.memory();

	print_size("hive usage", base);

	Block_p *blocks = new Block_p [OBJ_NUM];
	for (int i = 0; i < OBJ_NUM; ++i) {
		blocks[i] = Block_p(new Block(PROP_NUM));
		hive.register_object(blocks[i]);
	}
	next = hive.memory();
	print_size("object usage", (next - base) / OBJ_NUM);

	base = next;
	for (int i = 0; i < PROP_NUM; ++i) {
		hive.register_object(blocks[0], std::bind(&get_prop, std::placeholders::_1, i));
	}
	next = hive.memory();
	print_size("property usage", (next - base) / PROP_NUM);


	base = next;

	return 0;

#if 0
	tbb::tick_count t1, t2;


	int action_num = atoi(argv[1]);
	int actor_num = atoi(argv[2]);
	int prop_num = atoi(argv[3]);
	int obs_num = atoi(argv[4]);

	long obs_count = prop_num * obs_num;

	Block_p block = Block_p(new Block(prop_num));
	PropertyObs *observers = new PropertyObs [obs_count];
	PropertyAct *actors = new PropertyAct [actor_num];

	PVHive::PVHive &hive = PVHive::PVHive::get();






	std::cout << "# creating objects" << std::endl;
	t1 = tbb::tick_count::now();
	hive.register_object(block);
	for (int i = 0; i < prop_num; ++i) {
		hive.register_object(block, std::bind(&get_prop, std::placeholders::_1, i));
	}
	t2 = tbb::tick_count::now();
	print_stat("objects created", t1, t2, prop_num + 1);

	std::cout << "# creating properties" << std::endl;
	t1 = tbb::tick_count::now();
	hive.register_object(block);
	for (int i = 0; i < prop_num; ++i) {
		hive.register_object(block, std::bind(&get_prop, std::placeholders::_1, i));
	}
	t2 = tbb::tick_count::now();
	print_stat("properties created", t1, t2, prop_num + 1);


	std::cout << "# creating observers" << std::endl;
	t1 = tbb::tick_count::now();
	for (int i = 0; i < prop_num; ++i) {
		for (int j = 0; j < obs_num; ++j) {
			hive.register_observer(block, std::bind(&get_prop, std::placeholders::_1, i),
			                       observers[obs_num * i + j]);
		}
	}
	t2 = tbb::tick_count::now();
	print_stat("observers created", t1, t2, obs_count);


	std::cout << "# creating actors" << std::endl;
	t1 = tbb::tick_count::now();
	for (int i = 0; i < actor_num; ++i) {
		actors[i] = PropertyAct(rand() % prop_num);
		hive.register_actor(block, actors[i]);
	}
	t2 = tbb::tick_count::now();
	print_stat("actors created", t1, t2, actor_num);


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
#endif
}
