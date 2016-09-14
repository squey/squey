/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVSERIALIZEARCHIVEZIP_H
#define PVCORE_PVSERIALIZEARCHIVEZIP_H

#include <pvkernel/core/PVSerializeArchive.h>

namespace PVCore
{

class PVSerializeArchiveZip : public PVSerializeArchive
{
  public:
	PVSerializeArchiveZip(version_t v);
	PVSerializeArchiveZip(QString const& zip_path, archive_mode mode, version_t v);
	~PVSerializeArchiveZip() override;

  public:
	void open_zip(QString const& zip_path, archive_mode mode);
	void close_zip();

  protected:
	QString _zip_path;
	QString _tmp_path;
};
} // namespace PVCore

#endif
