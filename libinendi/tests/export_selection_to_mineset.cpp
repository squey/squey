/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include "common.h"

#include <inendi/PVMineset.h>

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}

	pvtest::TestEnv env(argv[1], argv[2]);
	env.compute_mapping();
	Inendi::PVView* view = env.compute_plotting()->get_parent<Inendi::PVRoot>()->current_view();

	std::string dataset_url = Inendi::PVMineset::import_dataset(*view);
	Inendi::PVMineset::delete_dataset(dataset_url);
}
