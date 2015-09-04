
#include <pvkernel/core/picviz_assert.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/common.h>

#include "common.h"

#include <stdlib.h>

#define FILENAME "zone_tree.dump"

typedef PVParallelView::PVZoneTree zt_t;

void clean()
{
	remove(FILENAME);
}

int main(int argc, char **argv)
{
	if ((argc != 2) && (argc != 4)) {
		usage(argv[0]);
		return 1;
	}

	atexit(clean);

	PVParallelView::common::init_cuda();

	PVParallelView::PVLibView* pv = create_lib_view_from_args(argc, argv);

	if (pv == nullptr) {
		return 1;
	}

	PVParallelView::PVZonesManager &zm = pv->get_zones_manager();

	for (PVZoneID zid = 0; zid < zm.get_number_of_managed_zones(); ++zid) {
		std::cout << "testing zone " << zid << std::endl;
		zt_t &zt = zm.get_zone_tree<zt_t>(zid);

		std::cout << "  dumping" << std::endl;
		bool ret = zt.dump_to_file(FILENAME);
		PV_VALID(ret, true);
		std::cout << "  done" << std::endl;

		std::cout << "  exhuming" << std::endl;
		zt_t *zt2 = zt_t::load_from_file(FILENAME);
		PV_ASSERT_VALID(zt2 != nullptr);
		std::cout << "  done" << std::endl;

		ret = (zt == *zt2);
		PV_VALID(ret, true);
		delete zt2;
	}

	return 0;
}
