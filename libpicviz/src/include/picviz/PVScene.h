//! \file PVScene.h
//! $Id: PVScene.h 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVSCENE_H
#define PICVIZ_PVSCENE_H

#include <QString>

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVInputType.h>
#include <picviz/PVPtrObjects.h>
#include <picviz/PVSource_types.h>

#include <boost/shared_ptr.hpp>

namespace Picviz {

/**
 * \class PVScene
 */
class LibPicvizDecl PVScene {
public:
	typedef boost::shared_ptr<PVScene> p_type;
	typedef QList<PVSource_p> list_sources_t;
public:
	
	PVScene(QString scene_name, PVRoot_p parent);
	~PVScene();

public:
	PVRoot_p get_root();

public:
	void add_input(PVRush::PVInputDescription_p in);
	void add_source(PVSource_p src);

protected:
	// PVRush::list_inputs is QList<PVRush::PVInputDescription_p>
	PVRush::PVInputType::list_inputs _inputs;
	list_sources_t _sources;

	PVRoot_p _root;
	QString _name;
};

typedef PVScene::p_type PVScene_p;

}

#endif	/* PICVIZ_PVSCENE_H */
