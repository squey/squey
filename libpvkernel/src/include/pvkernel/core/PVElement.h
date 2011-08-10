/*
 * $Id: PVElement.h 3221 2011-06-30 11:45:19Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVELEMENT_FILE_H
#define PVELEMENT_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVBufferSlice.h>

#include <list>

#include <tbb/scalable_allocator.h>

#include <QList>
#include <QExplicitlySharedDataPointer>

namespace PVCore {

class PVChunk;

class PVField;
class PVElementData;

typedef std::list<PVField, tbb::scalable_allocator<PVField> > list_fields;
class LibKernelDecl PVElement : public PVBufferSlice {
	friend class PVField;
	friend class PVChunk;
public:
	PVElement(PVChunk* parent, char* begin, char* end);
	PVElement(PVChunk* parent);
	PVElement(PVElement const& src);
	virtual ~PVElement();
public:
	bool valid() const;
	void set_invalid();
	void set_parent(PVChunk* parent);
	void save_elt_buffer();
	char* get_saved_elt_buffer(size_t& n);
	PVChunk* chunk_parent();
	chunk_index get_elt_index();
	chunk_index get_elt_agg_index();
	bool same_data_as(PVElement const& elt) const { return d == elt.d; }
	size_t get_chunk_index() const { return _chunk_index; }

	list_fields& fields();
	list_fields const& c_fields() const;

	buf_list_t& realloc_bufs() const;
public:
	PVElement& operator=(PVElement const& src);
protected:
	// Set by the parent PVChunk
	void set_chunk_index(size_t i) { _chunk_index = i; }
private:
	void clear_saved_buf();
	void init(PVChunk* parent);
protected:
	QExplicitlySharedDataPointer<PVElementData> d;
private:
	size_t _chunk_index;
};

}

#endif
