#include <pvkernel/rush/Normalize>

#include <picviz/nraw.h>
#include <picviz/init.h>
#include <picviz/general.h>
#include <picviz/scene.h>
#include <picviz/datatreerootitem.h>
#include <picviz/nraw.h>

#include <stdlib.h>

using namespace PVRush;

picviz_nraw_t *picviz_nraw_from_pvrush(apr_pool_t *pool, QList<QStringList>qtnraw);

int main(int argc, char **argv)
{
	picviz_nraw_t *nraw;
	picviz_datatreerootitem_t *datatree;
	picviz_scene_t *scene;
	Normalize *normalize = new Normalize();	
	QList<QStringList >qt_nraw;

#include "test-env.h"

	picviz_init(argc, NULL);

	datatree = picviz_datatreerootitem_new();
	scene = picviz_scene_new(datatree, "default");
	qt_nraw = normalize->normalize("pcre:syslog", "test_petit.log");

	nraw = picviz_nraw_from_pvrush(scene->pool, qt_nraw);

	picviz_terminate();

	delete normalize;

	return 0;

}

