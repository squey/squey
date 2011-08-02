//! \file PVScene.h
//! $Id: PVScene.h 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVSCENE_H
#define PICVIZ_PVSCENE_H

#include <QString>

#include <pvkernel/core/general.h>
#include <picviz/PVPtrObjects.h>

#include <boost/shared_ptr.hpp>

namespace Picviz {

/**
 * \class PVScene
 */
class LibPicvizDecl PVScene {
public:
	typedef boost::shared_ptr<PVScene> p_type;
public:
	PVRoot_p root;
	QString name;
	
	PVScene(QString scene_name, PVRoot_p parent);
	~PVScene();

};

typedef PVScene::p_type PVScene_p;

}

#endif	/* PICVIZ_PVSCENE_H */
