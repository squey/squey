/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef INENDI_PVSOURCECREATOROPCUA_H
#define INENDI_PVSOURCECREATOROPCUA_H

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVSourceCreator.h>

namespace PVRush
{

class PVSourceCreatorOpcUa : public PVSourceCreator
{
  public:
	source_p create_source_from_input(PVInputDescription_p input) const override;
	QString supported_type() const override;
	QString name() const override;

	CLASS_REGISTRABLE(PVSourceCreatorOpcUa)
};
} // namespace PVRush

#endif // INENDI_PVSOURCECREATOROPCUA_H
