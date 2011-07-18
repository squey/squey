//! \file PVSelectionFilterScatterPlotSelectionSquare.h
//! $Id: PVSelectionFilterScatterPlotSelectionSquare.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVSELECTIONFILTERSCATTERPLOTSELECTIONSQUARE_H
#define PICVIZ_PVSELECTIONFILTERSCATTERPLOTSELECTIONSQUARE_H


#include <pvcore/general.h>

#include <picviz/PVSelection.h>
#include <picviz/PVSelectionFilter.h>

namespace Picviz {

/**
 * \class PVSelectionFilterScatterPlotSelectionSquare
 */
class LibPicvizDecl PVSelectionFilterScatterPlotSelectionSquare : public PVSelectionFilter {
public:
	PVSelectionFilterScatterPlotSelectionSquare(PVFilter::PVArgumentList const& l = PVSelectionFilterScatterPlotSelectionSquare::default_args());

public:
	virtual void operator()(PVSelection& in, PVSelection &out);

public:
	void set_x1_min(float value) {_args["x1_min"] = value;}
	void set_x1_max(float value) {_args["x1_max"] = value;}
	void set_x2_min(float value) {_args["x2_min"] = value;}
	void set_x2_max(float value) {_args["x2_max"] = value;}
	
	CLASS_FILTER(Picviz::PVSelectionFilterScatterPlotSelectionSquare)
};
}

#endif
