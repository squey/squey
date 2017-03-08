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

const std::string filename = TEST_FOLDER "/picviz/heat_line.csv";
const std::string fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

int main()
{
	PVParallelView::common::RAII_backend_init resources;

	TestEnv env(filename, fileformat);

	PVParallelView::PVLibView* pv = env.get_lib_view();

	PVParallelView::PVZonesManager& zm = pv->get_zones_manager();

	for (PVZoneID zid(0); zid < zm.get_number_of_managed_zones(); ++zid) {
		std::cout << "testing zone " << zid << std::endl;

		std::cout << "  initialization, it can take a while" << std::endl;
		zm.request_zoomed_zone(zid);
		zzt_t& zzt = zm.get_zoom_zone_tree(zid);
		std::cout << "  done" << std::endl;

		zzt.reset();
	}

	return 0;
}
