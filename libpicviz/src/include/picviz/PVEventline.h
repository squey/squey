//! \file PVEventline.h
//! $Id: PVEventline.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVEVENTLINE_H
#define PICVIZ_PVEVENTLINE_H

#include <pvcore/general.h>
#include <picviz/PVSelection.h>

namespace Picviz {

/**
 * \class PVEventline
 */
class LibExport PVEventline {
public:
	PVEventline(PVRow row_count);
public:
	int get_current_index() const;
	int get_first_index() const;
	float get_kth_slider_position(int k) const;
	int get_last_index() const;

	void set_current_index(int index);
	void set_first_index(int index);
	float set_kth_index_and_adjust_slider_position(int k, float x);
	void set_last_index(int index);

	void selection_A2A_filter(PVSelection &selection);
	void selection_A2B_filter(PVSelection &a, PVSelection &b);

	int get_row_count() const;
	void set_row_count(int row_count_) { row_count = row_count_; }

private:
	void *dtri;
	void *parent;

	int row_count;
	
	int first_index;
	int current_index;
	int last_index;
};

}


#endif
