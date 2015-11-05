/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdlib.h>

#include <inendi/PVRoot.h>
#include <inendi/PVScene.h>
#include <inendi/PVSource.h>

using Inendi::PVRoot;
using Inendi::PVScene;
using Inendi::PVSource;

using Inendi::PVRoot_p;
using Inendi::PVScene_p;
using Inendi::PVSource_p;

int main(void)
{

#include "test-env.h"

	PVRoot_sp root = Inendi::PVRoot::get_root_sp();

	PVScene_p scene = PVScene_p(new PVScene(QString("scene1"), root));

	PVSource source(scene);

	source.file_append("pcre:syslog", "logs/syslog");

	return 0;
}
