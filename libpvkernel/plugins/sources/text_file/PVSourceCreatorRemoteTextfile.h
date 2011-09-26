#ifndef PICVIZ_PVSOURCECREATORREMOTETEXTFILE_H
#define PICVIZ_PVSOURCECREATORTEMOTETEXTFILE_H

#include "PVSourceCreatorTextfile.h""

namespace PVRush {

class PVSourceCreatorRemoteTextfile: public PVSourceCreatorTextfile
{
public:
	QString supported_type() const;

	CLASS_REGISTRABLE(PVSourceCreatorRemoteTextfile)
};

}

#endif
