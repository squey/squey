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

	pvtest::TestEnv env(argv[1], argv[2], 1, pvtest::ProcessUntil::View);
	Inendi::PVView* view = env.root.current_view();

	std::string dataset_url = Inendi::PVMineset::import_dataset(*view);
	Inendi::PVMineset::delete_dataset(dataset_url);
}
