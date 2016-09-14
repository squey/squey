/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVSERIALIZEARCHIVEOPTIONS_H
#define PVCORE_PVSERIALIZEARCHIVEOPTIONS_H

#include <pvkernel/core/PVSerializeArchive.h> // for PVSerializeArchive, etc
#include <cstddef>                            // for size_t
#include <vector>                             // for vector
#include <pvkernel/core/PVSerializeObject.h>
#include <QString>
#include <QVariant>

namespace PVCore
{

class PVArgumentList;

class PVSerializeArchiveOptions : public PVSerializeArchive
{
	friend class PVSerializeObject;

  public:
	explicit PVSerializeArchiveOptions(version_t version) : PVSerializeArchive(version)
	{
		_mode = write;
	}
	PVSerializeArchiveOptions(const PVSerializeArchiveOptions& obj) = delete;
	~PVSerializeArchiveOptions() override {}

  public:
	bool must_write(PVSerializeObject const& parent, QString const& name);
	void include_all_files(bool inc);
	int does_include_all_files() const;

  protected:
	// Object create function
	PVSerializeObject_p create_object(QString const& name, PVSerializeObject* parent) override;
	// Attribute access functions, here empty
	void attribute_write(PVSerializeObject const&, QString const&, QVariant const&) override{};
	void attribute_read(PVSerializeObject&, QString const&, QVariant&, QVariant const&) override{};
	void list_attributes_write(PVSerializeObject const&,
	                           QString const&,
	                           std::vector<QVariant> const&) override{};
	void list_attributes_read(PVSerializeObject const&,
	                          QString const&,
	                          std::vector<QVariant>&) override{};
	void hash_arguments_write(PVSerializeObject const&,
	                          QString const&,
	                          PVArgumentList const&) override{};
	size_t buffer(PVSerializeObject const&, QString const&, void*, size_t n) override { return n; };
	void file(PVSerializeObject const&, QString const&, QString&) override{};
};
} // namespace PVCore

#endif
