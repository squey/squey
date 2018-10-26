/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSOURCECREATORSPLUNK_H
#define INENDI_PVSOURCECREATORSPLUNK_H

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVSourceCreator.h>

namespace PVRush
{

class PVSourceCreatorSplunk : public PVSourceCreator
{
  public:
	source_p create_source_from_input(PVInputDescription_p input) const override;
	QString supported_type() const override;
	bool pre_discovery(PVInputDescription_p input) const override;
	QString name() const override;

	CLASS_REGISTRABLE(PVSourceCreatorSplunk)
};

} // namespace PVRush

#endif // INENDI_PVSOURCECREATORSPLUNK_H
