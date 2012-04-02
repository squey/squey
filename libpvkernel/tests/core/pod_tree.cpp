#include <pvkernel/core/general.h>
#include <pvkernel/core/PVPODTree.h>
#include <pvkernel/core/picviz_bench.h>

#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <vector>

template <class T>
void dump_tree(T const& pod_tree)
{
	for (typename T::size_type b = 0; b < T::nbranches(); b++) {
		typename T::const_branch_iterator it,ite;
		ite = pod_tree.end_branch(b);
		std::cout << "Branch " << b << ": ";
		for (it = pod_tree.begin_branch(b); it != ite; it++) {
			std::cout << *it << " ";
		}
		std::cout << std::endl;
	}
}

int main()
{
	{
		PVCore::PVPODTree<uint32_t, uint32_t, 4> pod_tree(1287);
		PVCore::PVPODTree<uint32_t, uint32_t, 4> pod_tree2(1287);
		for (int i = 0; i < 1287; i++) {
			pod_tree.push(i%4, i);
			pod_tree2.push(i%4, i);
		}
		dump_tree(pod_tree);

		std::cout << "-------" << std::endl;
		for (int i = 0; i < 4; i++) {
			pod_tree.move_branch(i, 3-i, pod_tree2);
		}
		dump_tree(pod_tree);

	}
	{
		PVCore::PVPODTree<uint32_t, uint32_t, 1288> pod_tree(104);
	}
	{
		PVCore::PVPODTree<uint32_t, uint32_t, 1<<20>* pod_tree = new PVCore::PVPODTree<uint32_t, uint32_t, 1<<20>(104);
		delete pod_tree;
	}
	{
		PVCore::PVPODTree<uint32_t, uint32_t, 1<<20>* pod_tree = new PVCore::PVPODTree<uint32_t, uint32_t, 1<<20>(100000);
		delete pod_tree;
	}

	srand(time(NULL));
	const uint32_t nelts = 417945007;
	PVCore::PVPODTree<uint32_t, uint32_t, 1<<20>* pod_tree = new PVCore::PVPODTree<uint32_t, uint32_t, 1<<20>(nelts);

	PVLOG_INFO("Initialising random buffer...\n");
	std::vector<uint32_t> relts;
	relts.reserve(nelts);
	for (uint32_t i = 0; i < nelts; i++) {
		relts.push_back(rand()*2+(rand()&1));
	}
	PVLOG_INFO("Done !\nCreating tree...\n");

	BENCH_START(org);
	for (uint32_t i = 0; i < nelts; i++) {
		const uint32_t relt = relts[i];
		pod_tree->push(relt >> 12, relt);
	}
	BENCH_END_TRANSFORM(org, "tree creation", nelts, sizeof(uint32_t));
	PVLOG_DEBUG("Number of block used: %llu\n", pod_tree->number_blocks_used());
	pod_tree->dump_buf_stats();
	//pod_tree->dump_branch_stats();

	return 0;
}
