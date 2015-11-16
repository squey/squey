/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdlib.h>

#include <inendi/PVMapped.h>
#include <inendi/PVMapping.h>
#include <inendi/PVPlotting.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVRoot.h>
#include <inendi/PVScene.h>
#include <inendi/PVSource.h>

#include <iostream>

using Inendi::PVMapped;
using Inendi::PVMapping;
using Inendi::PVPlotted;
using Inendi::PVPlotting;
using Inendi::PVRoot;
using Inendi::PVScene;
using Inendi::PVSource;

using Inendi::PVMapped_p;
using Inendi::PVMapping_p;
using Inendi::PVPlotted_p;
using Inendi::PVPlotting_p;
using Inendi::PVRoot_p;
using Inendi::PVScene_p;
using Inendi::PVSource_p;

int main(void)
{

#include "test-env.h"

	PVRoot_sp root = Inendi::PVRoot::get_root_sp();
	PVScene_p scene = PVScene_p(new PVScene(QString("scene1"), root));
	PVSource_p source = PVSource_p(new PVSource(scene));
	source->file_append("pcre:syslog", "logs/syslog");

	PVMapping_p mapping = PVMapping_p(new PVMapping(source));
	PVMapped_p mapped = PVMapped_p(new PVMapped(mapping));
	std::cout << "#### MAPPED ####\n";
	mapped->to_csv("");

	Inendi::PVRoot::release();

	return 0;
}
