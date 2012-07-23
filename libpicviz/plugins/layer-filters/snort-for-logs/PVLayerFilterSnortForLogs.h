/**
 * \file PVLayerFilterSnortForLogs.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVLAYERFILTERSNORTFORLOGS_H
#define PICVIZ_PVLAYERFILTERSNORTFORLOGS_H

#include <pvkernel/core/PVPython.h>
#include <pvkernel/core/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterSnortForLogs
 */
class PVLayerFilterSnortForLogs : public PVLayerFilter {
private:
	boost::python::dict _python_own_namespace;
	PyThreadState* _python_thread;
	boost::python::list snort_rules;
	int rules_number;
public:
	PVLayerFilterSnortForLogs(PVCore::PVArgumentList const& l = PVLayerFilterSnortForLogs::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);

	CLASS_FILTER(Picviz::PVLayerFilterSnortForLogs)

};
}

#endif	/* PICVIZ_PVLAYERFILTERSNORTFORLOGS_H */

