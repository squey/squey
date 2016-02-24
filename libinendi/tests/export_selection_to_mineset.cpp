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

	QDir nraw_dir(QString::fromStdString(PVRush::PVNraw::default_tmp_path));
	if (!nraw_dir.exists()){
		nraw_dir.mkdir(QString::fromStdString(PVRush::PVNraw::default_tmp_path));
	}

	std::string dataset_url = Inendi::PVMineset::import_dataset(*env.view);
	Inendi::PVMineset::delete_dataset(dataset_url);
}
