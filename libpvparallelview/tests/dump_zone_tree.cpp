
#include <pvkernel/core/picviz_assert.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/common.h>

#include "common.h"

#include <stdlib.h>

typedef PVParallelView::PVZoneTree zt_t;

int main(int argc, char **argv)
{
	set_extra_param(2, "dump_name_format (zone-id|\"all\")");

	PVParallelView::common::init_cuda();

	PVParallelView::PVLibView* pv = create_lib_view_from_args(argc, argv);

	if (pv == nullptr) {
		return 1;
	}

	int pos = extra_param_start_at();
	const char *dump_format = argv[pos];
	char dump_filename[2048];

	if (strstr(dump_format, "%d") == NULL) {
		std::cerr << "dump_name_format must contains a '%d', for example: 'path/file%d.dump'"
		          << std::endl;
		return 1;
	}

	PVParallelView::PVZonesManager &zm = pv->get_zones_manager();
	PVZoneID zmin;
	PVZoneID zmax = zm.get_number_of_managed_zones();

	if (std::string("all") != argv[pos+1]) {
		zmin = atoi(argv[pos+1]);
		if ((zmin < 0) || (zmin >= zmax)) {
			std::cerr << "zone-id is out of range, max is " << zmax - 1 << std::endl;
			return 1;
		}
		zmax = zmin + 1;
	} else {
		zmin = 0;
	}

	for (PVZoneID zid = zmin; zid < zmax; ++zid) {
		zt_t &zt = zm.get_zone_tree<zt_t>(zid);
		snprintf(dump_filename, 2048, dump_format, zid);
		std::cout << "  dumping zone " << zid << std::endl;
		PV_ASSERT_VALID(zt.dump_to_file(dump_filename));
	}

	return 0;
}
