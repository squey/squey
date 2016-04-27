/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVPTROBJECTS_H
#define INENDI_PVPTROBJECTS_H

// Defines shared ptr with classes
// Used to break cyclic dependencies

#include <memory>

namespace Inendi
{

class PVRoot;
// typedef std::shared_ptr<PVRoot> PVRoot_p;

class PVSource;
// typedef std::shared_ptr<PVSource> PVSource_p;

class PVScene;
// typedef std::shared_ptr<PVScene> PVScene_p;

class PVPlotting;
// typedef std::shared_ptr<PVPlotting> PVPlotting_p;

class PVPlotted;
// typedef std::shared_ptr<PVPlotted> PVPlotted_p;

class PVMapping;
// typedef std::shared_ptr<PVMapping> PVMapping_p;

class PVMapped;
// typedef std::shared_ptr<PVMapped> PVMapped_p;
}

#endif
