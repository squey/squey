
#include <pvkernel/core/picviz_assert.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoomedZoneTree.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/common.h>

#include "common.h"

#include <stdlib.h>

#define FILENAME "zoomed_zone_tree.dump"

typedef PVParallelView::PVZoomedZoneTree zzt_t;

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

#ifdef PICVIZ_DEVELOPER_MODE
	PVParallelView::common::init_cuda();

	PVParallelView::PVLibView* pv = create_lib_view_from_args(argc, argv);

	if (pv == nullptr) {
		return 1;
	}

	PVParallelView::PVZonesManager &zm = pv->get_zones_manager();

	for (PVZoneID zid = 0; zid < zm.get_number_of_managed_zones(); ++zid) {
		std::cout << "testing zone " << zid << std::endl;

		std::cout << "  initialization, it can take a while" << std::endl;
		zm.request_zoomed_zone(zid);
		zzt_t &zzt = zm.get_zone_tree<zzt_t>(zid);
		std::cout << "  done" << std::endl;

		std::cout << "  dumping" << std::endl;
		bool ret = zzt.dump_to_file(FILENAME);
		PV_VALID(ret, true);
		std::cout << "  done" << std::endl;

		std::cout << "  exhuming" << std::endl;
		zzt_t *zzt2 = zzt_t::load_from_file(FILENAME);
		PV_ASSERT_VALID(zzt2 != nullptr);
		std::cout << "  done" << std::endl;

		ret = (zzt == *zzt2);
		PV_VALID(ret, true);

		zzt.reset();
		delete zzt2;
	}
#endif

	return 0;
}
