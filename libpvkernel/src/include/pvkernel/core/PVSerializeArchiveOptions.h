/**
 * \file PVSerializeArchiveOptions.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVSERIALIZEARCHIVEOPTIONS_H
#define PVCORE_PVSERIALIZEARCHIVEOPTIONS_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSerializeArchiveOptions_types.h>

namespace PVCore {

class LibKernelDecl PVSerializeArchiveOptions: public PVSerializeArchive
{
	friend class PVSerializeObject;
public:
	typedef PVSerializeArchiveOptions_p p_type;
public:
	PVSerializeArchiveOptions(version_t version):
		PVSerializeArchive(version)
	{
		_mode = write;
	}
	virtual ~PVSerializeArchiveOptions() { this->PVSerializeArchive::~PVSerializeArchive(); }

private:
	PVSerializeArchiveOptions(const PVSerializeArchiveOptions& obj):
		PVSerializeArchive(obj)
	{ assert(false); }

public:
	bool must_write(PVSerializeObject const& parent, QString const& name);
	void include_all_files(bool inc);
	int does_include_all_files() const;

protected:
	// Object create function
	virtual PVSerializeObject_p create_object(QString const& name, PVSerializeObject* parent);
	// Attribute access functions, here empty
	virtual void attribute_write(PVSerializeObject const&, QString const&, QVariant const&) { };
	virtual void attribute_read(PVSerializeObject&, QString const&, QVariant&, QVariant const&) { };
	virtual void list_attributes_write(PVSerializeObject const&, QString const&, std::vector<QVariant> const&) { };
	virtual void list_attributes_read(PVSerializeObject const&, QString const&, std::vector<QVariant>&) { };
	virtual void hash_arguments_write(PVSerializeObject const&, QString const&, PVArgumentList const&) { };
	virtual void hash_arguments_read(PVSerializeObject const&, QString const&, PVArgumentList&) { };
	virtual size_t buffer(PVSerializeObject const&, QString const&, void*, size_t n) { return n; };
	virtual void file(PVSerializeObject const&, QString const&, QString&) { };
};

}

#endif
