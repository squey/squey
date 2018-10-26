/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSOURCECREATORTEXTFILE_H
#define INENDI_PVSOURCECREATORTEXTFILE_H

/* #include <pvkernel/core/PVArgument.h> */
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVUnicodeSource.h>

namespace PVRush
{

class PVSourceCreatorTextfile : public PVSourceCreator
{
  public:
	source_p create_source_from_input(PVInputDescription_p input) const override;
	QString supported_type() const override;
	bool pre_discovery(PVInputDescription_p input) const override;
	QString name() const override;

	CLASS_REGISTRABLE(PVSourceCreatorTextfile)
};
}

#endif
