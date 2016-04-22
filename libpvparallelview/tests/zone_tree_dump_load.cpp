/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_assert.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/common.h>

#include "common.h"

constexpr const char* dump_file = "/tmp/zone_tree.dump";
const std::string filename = TEST_FOLDER "/picviz/heat_line.csv";
const std::string fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

typedef PVParallelView::PVZoneTree zt_t;

void clean()
{
	remove(dump_file);
}

int main()
{
	atexit(clean);

	PVParallelView::common::RAII_cuda_init cuda_resources;

	TestEnv env(filename, fileformat);

	PVParallelView::PVLibView* pv = env.get_lib_view();

	PVParallelView::PVZonesManager &zm = pv->get_zones_manager();

	for (PVZoneID zid = 0; zid < zm.get_number_of_managed_zones(); ++zid) {
		std::cout << "testing zone " << zid << std::endl;
		zt_t &zt = zm.get_zone_tree(zid);

		std::cout << "  dumping" << std::endl;
		bool ret = zt.dump_to_file(dump_file);
		PV_VALID(ret, true);
		std::cout << "  done" << std::endl;

		std::cout << "  exhuming" << std::endl;
		zt_t *zt2 = zt_t::load_from_file(dump_file);
		PV_ASSERT_VALID(zt2 != nullptr);
		std::cout << "  done" << std::endl;

		ret = (zt == *zt2);
		PV_VALID(ret, true);
		delete zt2;
	}

	return 0;
}
