
#include <pvkernel/core/picviz_assert.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVZoomedZoneTree.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/common.h>

#include "common.h"

#include <stdlib.h>

typedef PVParallelView::PVZoneTree zt_t;
typedef PVParallelView::PVZoomedZoneTree zzt_t;

int main(int argc, char **argv)
{
	set_extra_param(2, "zt_dump_name_format zzt_dump_name_format");

	PVParallelView::common::init_cuda();

	PVParallelView::PVLibView* pv = create_lib_view_from_args(argc, argv);

	if (pv == nullptr) {
		return 1;
	}

	int pos = extra_param_start_at();

	const char *zt_dump_format = nullptr;
	char zt_dump_filename[2048];

	if (strlen(argv[pos]) != 0) {
		zt_dump_format = argv[pos];
		if (strstr(zt_dump_format, "%d") == NULL) {
			std::cerr << "zt_dump_name_format must contains a '%d', for example: 'path/file%d.dump'"
			          << std::endl;
			return 1;
		}
	}

	const char *zzt_dump_format = nullptr;
	char zzt_dump_filename[2048];

	if (strlen(argv[pos + 1]) != 0) {
		zzt_dump_format = argv[pos + 1];
		if (strstr(zzt_dump_format, "%d") == NULL) {
			std::cerr << "zzt_dump_name_format must contains a '%d', for example: 'path/file%d.dump'"
			          << std::endl;
			return 1;
		}
	}

	PVParallelView::PVZonesManager &zm = pv->get_zones_manager();

	if (zt_dump_format != nullptr) {
		for (PVZoneID zid = 0; zid < zm.get_number_of_managed_zones(); ++zid) {
			zt_t &zt = zm.get_zone_tree<zt_t>(zid);
			snprintf(zt_dump_filename, 2048, zt_dump_format, zid);
			std::cout << "dumping zone " << zid << std::endl;
			PV_ASSERT_VALID(zt.dump_to_file(zt_dump_filename));
		}
	}


	if (zzt_dump_format != nullptr) {
		for (PVZoneID zid = 0; zid < zm.get_number_of_managed_zones(); ++zid) {
			zzt_t &zzt = zm.get_zone_tree<zzt_t>(zid);
			snprintf(zzt_dump_filename, 2048, zzt_dump_format, zid);
			std::cout << "dumping zoomed zone " << zid << std::endl;
			PV_ASSERT_VALID(zzt.dump_to_file(zzt_dump_filename));
			zzt.reset();
		}
	}

	return 0;
}
