/**
 * \file PVPtrObjects.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVPTROBJECTS_H
#define PICVIZ_PVPTROBJECTS_H

// Defines shared ptr with classes
// Used to break cyclic dependencies

#include <memory>

namespace Picviz {

class PVRoot;
//typedef std::shared_ptr<PVRoot> PVRoot_p;

class PVSource;
//typedef std::shared_ptr<PVSource> PVSource_p;

class PVScene;
//typedef std::shared_ptr<PVScene> PVScene_p;

class PVPlotting;
//typedef std::shared_ptr<PVPlotting> PVPlotting_p;

class PVPlotted;
//typedef std::shared_ptr<PVPlotted> PVPlotted_p;

class PVMapping;
//typedef std::shared_ptr<PVMapping> PVMapping_p;

class PVMapped;
//typedef std::shared_ptr<PVMapped> PVMapped_p;

}

#endif
