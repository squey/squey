/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_assert.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoomedZoneTree.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/common.h>

#include "common.h"


using zzt_t = PVParallelView::PVZoomedZoneTree;

const std::string dump_file = "/tmp/zoomed_zone_tree.dump";
const std::string filename = TEST_FOLDER "/picviz/heat_line.csv";
const std::string fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

void clean()
{
	remove(dump_file.c_str());
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

		std::cout << "  initialization, it can take a while" << std::endl;
		zm.request_zoomed_zone(zid);
		zzt_t &zzt = zm.get_zoom_zone_tree(zid);
		std::cout << "  done" << std::endl;

		std::cout << "  dumping" << std::endl;
		bool ret = zzt.dump_to_file(dump_file.c_str());
		PV_VALID(ret, true);
		std::cout << "  done" << std::endl;

		std::cout << "  exhuming" << std::endl;
		zzt_t *zzt2 = zzt_t::load_from_file(dump_file.c_str());
		PV_ASSERT_VALID(zzt2 != nullptr);
		std::cout << "  done" << std::endl;

		ret = (zzt == *zzt2);
		PV_VALID(ret, true);

		zzt.reset();
		delete zzt2;
	}

	return 0;
}
