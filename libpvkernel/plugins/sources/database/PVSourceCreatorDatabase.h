/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSOURCECREATORDATABASE_H
#define INENDI_PVSOURCECREATORDATABASE_H

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVSourceCreator.h>

namespace PVRush
{

class PVSourceCreatorDatabase : public PVSourceCreator
{
  public:
	source_p create_source_from_input(PVInputDescription_p input) const override;
	QString supported_type() const override;
	QString name() const override;

	CLASS_REGISTRABLE(PVSourceCreatorDatabase)
};
}

#endif
