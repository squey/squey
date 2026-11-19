/* * MIT License
 *
 * © Squey, 2024
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVPARQUETDESCRIPTION_H__
#define __PVPARQUETDESCRIPTION_H__

#include <pvkernel/rush/PVFileDescription.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/pointer.h>

#include <QDir>
#include <QFileInfo>
#include <QSet>
#include <QSettings>
#include <QStringList>

#include <memory>
#include <string>
#include <vector>

namespace PVRush
{

class PVParquetFileDescription : public PVFileDescription
{
  public:
	PVParquetFileDescription(
		QStringList const& paths
	)
	    : PVFileDescription(paths.front(), paths.size() > 1 /*multi_inputs*/)
		, _paths(paths)
	{
	}

  public:
  	const QStringList& paths() const { return _paths; }

	void disable_multi_inputs(bool disabled) { _multi_inputs = not disabled; }

	/**
	 * The names of the files: their file names, or, when two of them share one, their paths from
	 * the deepest directory holding them all.
	 */
	QStringList input_names() const
	{
		QStringList names;
		for (const QString& path : _paths) {
			names << QFileInfo(path).fileName();
		}
		if (QSet<QString>(names.begin(), names.end()).size() == names.size()) {
			return names;
		}

		QStringList absolute_paths;
		for (const QString& path : _paths) {
			absolute_paths << QDir::cleanPath(QFileInfo(path).absoluteFilePath());
		}

		QStringList common_dirs = QFileInfo(absolute_paths.front()).path().split('/');
		for (const QString& path : absolute_paths) {
			const QStringList dirs = QFileInfo(path).path().split('/');
			qsizetype shared = 0;
			while (shared < common_dirs.size() and shared < dirs.size() and
			       common_dirs[shared] == dirs[shared]) {
				shared++;
			}
			common_dirs.resize(shared);
		}
		// Files on different Windows drives have no directory in common.
		if (common_dirs.isEmpty()) {
			return absolute_paths;
		}
		// An absolute path starts with an empty name, before its first separator.
		const QDir common_dir(common_dirs == QStringList{""} ? "/" : common_dirs.join('/'));

		names.clear();
		for (const QString& path : absolute_paths) {
			names << common_dir.relativeFilePath(path);
		}
		return names;
	}

  private:
  	QStringList _paths;

  public:

	bool operator==(const PVInputDescription& other) const override
	{
		auto const* o = dynamic_cast<const PVParquetFileDescription*>(&other);
		return o != nullptr and _paths == o->_paths;
	}

	void save_to_qsettings(QSettings& settings) const override
	{
		settings.setValue("paths", _paths);
	}

	static std::vector<std::string> desc_from_qsetting(QSettings const& s)
	{
		QStringList paths = s.value("paths").toStringList();
		if (paths.isEmpty()) { // repli sur l'ancienne clé mono-fichier
			paths << s.value("path").toString();
		}
		std::vector<std::string> res;
		res.reserve(paths.size());
		for (const QString& path : paths) {
			res.push_back(path.toStdString());
		}
		return res;
	}

	static std::unique_ptr<PVInputDescription> load_from_string(pvcop::db::format const& fields,
	                                                            bool /*multi_inputs*/)
	{
		QStringList paths;
		for (const std::string& field : fields) {
			paths << QString::fromStdString(field);
		}
		return std::unique_ptr<PVParquetFileDescription>(new PVParquetFileDescription(paths));
	}

  public:
	void serialize_write(PVCore::PVSerializeObject& so) const override
	{
		so.set_current_status("Saving source file information...");
		so.attribute_write("files_path", _paths);
	}

	static std::unique_ptr<PVInputDescription> serialize_read(PVCore::PVSerializeObject& so)
	{
		so.set_current_status("Loading source file information...");
		QStringList paths = so.attribute_read<QStringList>("files_path");

		// File exists, continue with it
		for (const QString& path : paths) {
			if (not QFileInfo(path).isReadable()) {
				throw PVCore::PVSerializeReparaibleFileError(
					"Source file: '" + path.toStdString() + "' can't be found",
					so.get_logical_path().toStdString(), path.toStdString());
			}
		}

		return std::unique_ptr<PVInputDescription>(new PVParquetFileDescription(paths));
	}
};

} // namespace PVRush

#endif // __PVPARQUETDESCRIPTION_H__
