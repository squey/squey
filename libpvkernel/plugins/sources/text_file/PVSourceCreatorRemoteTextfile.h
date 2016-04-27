/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSOURCECREATORREMOTETEXTFILE_H
#define INENDI_PVSOURCECREATORREMOTETEXTFILE_H

#include "PVSourceCreatorTextfile.h"

namespace PVRush
{

class PVSourceCreatorRemoteTextfile : public PVSourceCreatorTextfile
{
  public:
	QString supported_type() const;

	CLASS_REGISTRABLE(PVSourceCreatorRemoteTextfile)
};
}

#endif
