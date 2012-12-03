/**
 * \file massive-mem.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <pvkernel/core/picviz_bench.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>

#include "massive_common.h"

#include <pvkernel/core/picviz_stat.h>

#if 0
void print_size(const char *text, size_t size, int num)
{
	static const char *units[] = {"o", "Kio", "Mio", "Gio", "Tio", "Pio"};
	int gindex = 0;
	int eindex = 0;
	uint64_t gscale = 1;
	uint64_t escale = 1;

	size_t gmem = size;
	size_t gs = gmem;

	size_t emem = size / num;
	size_t es = emem;

	while (gs > 1024) {
		gscale *= 1024;
		++gindex;
		gs /= 1024;
	}

	while (es > 1024) {
		escale *= 1024;
		++eindex;
		es /= 1024;
	}

	double dgmem = (double)gmem / gscale;
	double demem = (double)emem / escale;

	printf("%s: %g %s for %d element(s) -> %g %s per element\n",
	       text, dgmem, units[gindex], num, demem, units[eindex]);
}
#endif

class Block2
{
public:
	Block2()
	{}

private:
	int _i;
};

typedef PVCore::PVSharedPtr<Block2> Block2_p;

/*****************************************************************************
 * main
 *****************************************************************************/

int main(int argc, char **argv)
{
	if (argc <= 1) {
		std::cerr << "usage: " << argv[0] << " elements_number" << std::endl;
		return 1;
	}

	int element_num = atoi(argv[1]);

	size_t base, next;

	PVHive::PVHive &hive = PVHive::PVHive::get();
	base = hive.memory();
	std::cout << "hive usage: " << base << " o" << std::endl;
	PV_STAT_MEM_USE_O("hive", base);
	std::cout << std::endl;

	Block2_p *blocks2 = new Block2_p [element_num];
	next = base;
	{
		MEM_START(blk);
		for (int i = 0; i < element_num; ++i) {
			blocks2[i] = Block2_p(new Block2());
		}
		MEM_END(blk, "allocated Block2_p");
		std::cout << std::endl;
	}

	{
		MEM_START(obj);
		for (int i = 0; i < element_num; ++i) {
			hive.register_object(blocks2[i]);
		}
		MEM_END(obj, "registered objects");
	}
	next = hive.memory();
	// print_size("object usage", next - base, element_num);
	PV_STAT_MEM_USE_O("object", (next - base) / element_num);
	std::cout << std::endl;


	Block_p block = Block_p(new Block(element_num));
	base = next;
	{
		MEM_START(prop);
		for (int i = 0; i < element_num; ++i) {
			hive.register_object(block, std::bind(&get_prop, std::placeholders::_1, i));
		}
		MEM_END(prop, "registered properties");
	}
	next = hive.memory();
	// print_size("property usage", next - base, element_num);
	PV_STAT_MEM_USE_O("property", (next - base) / element_num);
	std::cout << std::endl;


	BlockObs *block_obs = new BlockObs [element_num];
	base = next;
	{
		MEM_START(objobs);
		for (int i = 0; i < element_num; ++i) {
			hive.register_observer(block, block_obs[i]);
		}
		MEM_END(objobs, "registered objects observers");
	}
	next = hive.memory();
	// print_size("object observer usage", next - base, element_num);
	PV_STAT_MEM_USE_O("object_observer", (next - base) / element_num);
	std::cout << std::endl;


	PropertyObs *prop_obs = new PropertyObs [element_num];
	base = next;
	{
		MEM_START(propobs);
		for (int i = 0; i < element_num; ++i) {
			hive.register_observer(block, std::bind(&get_prop, std::placeholders::_1, 0),
			                       prop_obs[i]);
		}
		MEM_END(propobs, "registered properties observers");
	}
	next = hive.memory();
	// print_size("property observer usage", next - base, element_num);
	PV_STAT_MEM_USE_O("property_observer", (next - base) / element_num);
	std::cout << std::endl;


	BlockAct *block_act = new BlockAct [element_num];
	base = next;
	{
		MEM_START(objact);
		for (int i = 0; i < element_num; ++i) {
			hive.register_actor(block, block_act[i]);
		}
		MEM_END(objact, "registered objects actors");
	}
	next = hive.memory();
	// print_size("object actor usage", next - base, element_num);
	PV_STAT_MEM_USE_O("object_actor", (next - base) / element_num);
	std::cout << std::endl;


	PropertyAct *prop_act = new PropertyAct [element_num];
	base = next;
	{
		MEM_START(propact);
		for (int i = 0; i < element_num; ++i) {
			hive.register_actor(block, prop_act[i]);
		}
		MEM_END(propact, "registered properties actors");
	}
	next = hive.memory();
	// print_size("property actor usage", next - base, element_num);
	PV_STAT_MEM_USE_O("property_actor", (next - base) / element_num);
	std::cout << std::endl;

	return 0;
}
