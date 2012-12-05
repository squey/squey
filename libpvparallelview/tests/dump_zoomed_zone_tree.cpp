
#include <pvkernel/core/picviz_assert.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoomedZoneTree.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/common.h>

#include "common.h"

#include <stdlib.h>

typedef PVParallelView::PVZoomedZoneTree zzt_t;

int main(int argc, char **argv)
{
	set_extra_param(1, "dump_name_format");

	PVParallelView::common::init_cuda();

	PVParallelView::PVLibView* pv = create_lib_view_from_args(argc, argv);

	if (pv == nullptr) {
		return 1;
	}

	const char *dump_format = argv[extra_param_start_at()];
	char dump_filename[2048];

	if (strstr(dump_format, "%d") == NULL) {
		std::cerr << "dump_name_format must contains a '%d', for example: 'path/file%d.dump'"
		          << std::endl;
		return 1;
	}

	PVParallelView::PVZonesManager &zm = pv->get_zones_manager();

	for (PVZoneID zid = 0; zid < zm.get_number_of_managed_zones(); ++zid) {
		std::cout << "processing zone " << zid << std::endl;

		std::cout << "  creation..." << std::endl;
		zm.request_zoomed_zone(zid);
		zzt_t &zzt = zm.get_zone_tree<zzt_t>(zid);
		std::cout << "  done" << std::endl;

		snprintf(dump_filename, 2048, dump_format, zid);
		std::cout << "  dumping to " << dump_filename << "..." << std::endl;
		PV_ASSERT_VALID(zzt.dump_to_file(dump_filename));
		std::cout << "  done" << std::endl;

		zzt.reset();
	}

	return 0;
}
