//! \file PVLayerFilterEntropy.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYERFILTERENTROPY_H
#define PICVIZ_PVLAYERFILTERENTROPY_H

#include <pvkernel/core/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterEntropy
 */
class PVLayerFilterEntropy : public PVLayerFilter {
private:
	int rules_number;
public:
	PVLayerFilterEntropy(PVCore::PVArgumentList const& l = PVLayerFilterEntropy::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);
	virtual QList<PVCore::PVArgumentKey> get_args_keys_for_preset() const;
	virtual QString menu_name() const { return "Entropy"; }

	CLASS_FILTER(Picviz::PVLayerFilterEntropy)

};
}

#endif	/* PICVIZ_PVLAYERFILTERENTROPY_H */

