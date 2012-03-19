//! \file PVAxis.h
//! $Id: PVAxis.h 2590 2011-05-07 15:43:12Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVAXIS_H
#define PICVIZ_PVAXIS_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVAxisFormat.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVAxis
 */
class LibPicvizDecl PVAxis: public PVRush::PVAxisFormat {
public:
	bool is_expandable;
	bool is_expanded;
	float thickness;

	/**
	 * Constructor
	 */
	PVAxis();
	PVAxis(PVRush::PVAxisFormat const& axis_format);

	/**
	 * Destructor
	 */
	~PVAxis();

public:
	PVCore::PVArgumentList const& get_args_mapping() const { return _args_mapping; }
	PVCore::PVArgumentList const& get_args_plotting() const { return _args_plotting; }
	
private:
	void init();
	static PVCore::PVArgumentList args_from_node(node_args_t const& args_str, PVCore::PVArgumentList const& def_args);

private:
	PVCore::PVArgumentList _args_mapping;
	PVCore::PVArgumentList _args_plotting;
};
}

#endif	/* PICVIZ_PVAXIS_H */
