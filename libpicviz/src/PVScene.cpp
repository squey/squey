//! \file PVScene.cpp
//! $Id: PVScene.cpp 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>

/******************************************************************************
 *
 * Picviz::PVScene::PVScene
 *
 *****************************************************************************/
Picviz::PVScene::PVScene(QString scene_name, PVRoot_p parent)
{
	root = parent;
	name = scene_name;

	// if (parent) {
	// 	parent->scene_append(this);
	// }
}

/******************************************************************************
 *
 * Picviz::PVScene::~PVScene
 *
 *****************************************************************************/
Picviz::PVScene::~PVScene()
{

}
