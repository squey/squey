/**
 * \file PVLayerFilterEntropy.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

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
	virtual QString status_bar_description();
	virtual QString detailed_description();

	CLASS_FILTER(Picviz::PVLayerFilterEntropy)

};
}

#endif	/* PICVIZ_PVLAYERFILTERENTROPY_H */

