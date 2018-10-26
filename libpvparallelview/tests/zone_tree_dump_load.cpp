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

const std::string filename = TEST_FOLDER "/picviz/heat_line.csv";
const std::string fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

typedef PVParallelView::PVZoneTree zt_t;

int main()
{
	PVParallelView::common::RAII_backend_init resources;

	TestEnv env(filename, fileformat);

	PVParallelView::PVLibView* pv = env.get_lib_view();

	PVParallelView::PVZonesManager& zm = pv->get_zones_manager();

	for (size_t zone_index(0); zone_index < zm.get_number_of_zones(); ++zone_index) {
		PVZoneID zid = zm.get_zone_id(zone_index);
		std::cout << "testing zone " << zid << std::endl;
		zm.get_zone_tree(zid);
	}

	return 0;
}
