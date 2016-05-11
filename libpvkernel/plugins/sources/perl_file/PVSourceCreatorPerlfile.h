/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSOURCECREATORPERLFILE_H
#define INENDI_PVSOURCECREATORPERLFILE_H

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVSourceCreator.h>

#include "PVPerlSource.h"

namespace PVRush
{

class PVSourceCreatorPerlfile : public PVSourceCreator
{
  public:
	source_p create_source_from_input(PVInputDescription_p input, const PVFormat& format) const;
	QString supported_type() const;
	hash_formats get_supported_formats() const;
	bool pre_discovery(PVInputDescription_p input) const;
	QString name() const;

	CLASS_REGISTRABLE(PVSourceCreatorPerlfile)
};
}

#endif /* INENDI_PVSOURCECREATORPERLFILE_H */
