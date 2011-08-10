/*
 * $Id: PVField.h 3221 2011-06-30 11:45:19Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVFIELD_FILE_H
#define PVFIELD_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVBufferSlice.h>
#include <pvkernel/core/PVElement.h>

namespace PVCore {

class LibKernelDecl PVField : public PVBufferSlice {
public:
	PVField(PVElement const& parent, char* begin, char* end);
	PVField(PVElement const& parent);
public:
	bool valid() const;
	void set_invalid();
	PVElement* elt_parent();
	void set_parent(PVElement const& parent);
	void deep_copy();
	size_t get_index_of_parent_element();
	size_t get_agg_index_of_parent_element();
private:
	void init(PVElement const& parent);
protected:
	bool _valid;
	PVElementData* _parent;
};

}

#endif
