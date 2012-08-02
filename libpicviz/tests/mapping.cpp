/**
 * \file mapping.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <stdlib.h>

#include <picviz/PVMapped.h>
#include <picviz/PVMapping.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>

#include <iostream>

using Picviz::PVMapped;
using Picviz::PVMapping;
using Picviz::PVPlotted;
using Picviz::PVPlotting;
using Picviz::PVRoot;
using Picviz::PVScene;
using Picviz::PVSource;

using Picviz::PVMapped_p;
using Picviz::PVMapping_p;
using Picviz::PVPlotted_p;
using Picviz::PVPlotting_p;
using Picviz::PVRoot_p;
using Picviz::PVScene_p;
using Picviz::PVSource_p;

int main(void)
{

#include "test-env.h"

	PVRoot_p root = PVRoot_p(new PVRoot());
	PVScene_p scene = PVScene_p(new PVScene(QString("scene1"), root));
	PVSource_p source = PVSource_p(new PVSource(scene));
	source->file_append("pcre:syslog", "logs/syslog");

	PVMapping_p mapping = PVMapping_p(new PVMapping(source));
	PVMapped_p mapped = PVMapped_p(new PVMapped(mapping));
	std::cout << "#### MAPPED ####\n";
	mapped->to_csv("");

	return 0;
}
