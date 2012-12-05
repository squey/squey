
#include <pvkernel/core/picviz_assert.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZoomedZoneTree.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/common.h>

#include "common.h"

#include <libgen.h>

#include <sstream>

#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_stat.h>

typedef PVParallelView::PVZoomedZoneTree zzt_t;

int main(int argc, char **argv)
{
	std::stringstream ss;

	PVParallelView::common::init_cuda();

	PVParallelView::PVLibView* pv = create_lib_view_from_args(argc, argv);

	if (pv == nullptr) {
		return 1;
	}

	ss << "load_";
	if (input_is_a_file()) {
		ss << basename(argv[1]);
	} else {
		ss << argv[2] << "_" << argv[3];
	}
	std::string stat_prefix = ss.str();

	PVParallelView::PVZonesManager &zm = pv->get_zones_manager();

	for (PVZoneID zid = 0; zid < zm.get_number_of_managed_zones(); ++zid) {
		ss.str("");
		ss << stat_prefix << "_" << zid;
		std::cout << "processing zone " << zid << std::endl;

		BENCH_START(load);
		zm.request_zoomed_zone(zid);
		BENCH_STOP(load);

		PV_STAT_TIME_SEC(ss.str(), BENCH_END_TIME(load));

		zm.get_zone_tree<zzt_t>(zid).reset();
	}

	return 0;
}
