#ifndef PICVIZ_PVPTROBJECTS_H
#define PICVIZ_PVPTROBJECTS_H

// Defines shared ptr with classes
// Used to break cyclic dependencies

#include <boost/shared_ptr.hpp>

namespace Picviz {

class PVRoot;
//typedef boost::shared_ptr<PVRoot> PVRoot_p;

class PVSource;
//typedef boost::shared_ptr<PVSource> PVSource_p;

class PVScene;
//typedef boost::shared_ptr<PVScene> PVScene_p;

class PVPlotting;
//typedef boost::shared_ptr<PVPlotting> PVPlotting_p;

class PVPlotted;
//typedef boost::shared_ptr<PVPlotted> PVPlotted_p;

class PVMapping;
//typedef boost::shared_ptr<PVMapping> PVMapping_p;

class PVMapped;
//typedef boost::shared_ptr<PVMapped> PVMapped_p;

}

#endif
