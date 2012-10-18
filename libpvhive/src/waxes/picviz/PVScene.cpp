/**
 * \file PVScene.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvhive/PVHive.h>
#include <pvhive/waxes/picviz/PVScene.h>

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

IMPL_WAX(Picviz::PVScene::select_view, scene, args)
{
	Picviz::PVView* old_cur_view = scene->current_view();
	call_object_default<Picviz::PVScene, FUNC(Picviz::PVScene::select_view)>(scene, args);
	if (old_cur_view) {
		refresh_observers(old_cur_view);
	}
	refresh_observers(&std::get<0>(args));
}

PVHIVE_CALL_OBJECT_BLOCK_END()
