/**
 * \file source.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <stdlib.h>

#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>

using Picviz::PVRoot;
using Picviz::PVScene;
using Picviz::PVSource;

using Picviz::PVRoot_p;
using Picviz::PVScene_p;
using Picviz::PVSource_p;

int main(void)
{

#include "test-env.h"

	PVRoot_sp root = Picviz::PVRoot::get_root_sp();

	PVScene_p scene = PVScene_p(new PVScene(QString("scene1"), root));

	PVSource source(scene);

	source.file_append("pcre:syslog", "logs/syslog");

	return 0;
}
