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
Picviz::PVScene::PVScene(QString scene_name, PVRoot_p parent):
	_root(parent),
	_name(scene_name)
{
}

/******************************************************************************
 *
 * Picviz::PVScene::~PVScene
 *
 *****************************************************************************/
Picviz::PVScene::~PVScene()
{
}

void Picviz::PVScene::add_input(PVRush::PVInputDescription_p in)
{
	_inputs.push_back(in);
}

void Picviz::PVScene::add_source(PVSource_p src)
{
	_sources.push_back(src);
}

Picviz::PVRoot_p Picviz::PVScene::get_root()
{
	return _root;
}
