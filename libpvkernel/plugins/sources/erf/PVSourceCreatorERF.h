/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#ifndef INENDI_PVSOURCECREATORERF_H
#define INENDI_PVSOURCECREATORERF_H

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVUnicodeSource.h>

namespace PVRush
{

class PVSourceCreatorERF : public PVRush::PVSourceCreator
{
  public:
	source_p create_source_from_input(PVRush::PVInputDescription_p input) const override;
	QString supported_type() const override;
	bool pre_discovery(PVRush::PVInputDescription_p input) const override;
	bool custom_multi_inputs() const override { return true; }
	QString name() const override;

	CLASS_REGISTRABLE(PVSourceCreatorERF)
};
} // namespace PVRush

#endif // INENDI_PVSOURCECREATORERF_H
