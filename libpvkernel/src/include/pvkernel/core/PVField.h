/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFIELD_FILE_H
#define PVFIELD_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVBufferSlice.h>
#include <pvkernel/core/PVDecimalStorage.h>

namespace PVCore {

class PVElement;

class PVField : public PVBufferSlice {
public:
	PVField(PVElement& parent, char* begin, char* end);
	PVField(PVElement& parent);
public:
	bool valid() const;
	void set_invalid();
	PVElement* elt_parent();
	void set_parent(PVElement& parent);
	void deep_copy();
	size_t get_index_of_parent_element();
	size_t get_agg_index_of_parent_element();
	inline operator QString() const { QString ret; get_qstr(ret); return ret; }
private:
	void init(PVElement& parent);
protected:
	bool _valid;
	PVElement* _parent;
};

}

#endif
