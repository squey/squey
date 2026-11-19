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

#ifndef __PVRUSH_PVFILELISTDESCRIPTION_H__
#define __PVRUSH_PVFILELISTDESCRIPTION_H__

#include <pvkernel/rush/PVFileDescription.h>

#include <QDir>
#include <QFileInfo>
#include <QSet>
#include <QSettings>
#include <QStringList>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace PVRush
{

/**
 * Files that a source reads one after the other, such as Parquet files. A source of several of
 * them starts with a column naming the file of each row, unless disable_multi_inputs() says
 * otherwise.
 *
 * The description keeps which columns and rows of the files to import, for a source read again,
 * from an investigation or a recent item, to hold what it held.
 */
class PVFileListDescription : public PVFileDescription
{
  public:
	PVFileListDescription(
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

	/** The names of the columns to import, or none for all of them. */
	const QStringList& columns() const { return _columns; }
	void set_columns(QStringList columns) { _columns = std::move(columns); }

	/** The SQL condition that the rows to import hold, or none for all of the rows. */
	const QString& row_filter() const { return _row_filter; }
	void set_row_filter(QString row_filter) { _row_filter = std::move(row_filter); }

  private:
  	QStringList _paths;
	QStringList _columns;
	QString _row_filter;

  public:

	bool operator==(const PVInputDescription& other) const override
	{
		auto const* o = dynamic_cast<const PVFileListDescription*>(&other);
		return o != nullptr and _paths == o->_paths and _columns == o->_columns and
		       _row_filter == o->_row_filter;
	}

	void save_to_qsettings(QSettings& settings) const override
	{
		settings.setValue("paths", _paths);
		settings.setValue("columns", _columns);
		settings.setValue("row_filter", _row_filter);
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

		// After an empty field, which no path is: the row filter, then the columns.
		const QStringList columns = s.value("columns").toStringList();
		const QString row_filter = s.value("row_filter").toString();
		if (not columns.isEmpty() or not row_filter.isEmpty()) {
			res.emplace_back();
			res.push_back(row_filter.toStdString());
			for (const QString& column : columns) {
				res.push_back(column.toStdString());
			}
		}
		return res;
	}

	static std::unique_ptr<PVInputDescription> load_from_string(pvcop::db::format const& fields,
	                                                            bool /*multi_inputs*/)
	{
		auto description = std::make_unique<PVFileListDescription>(paths_of(fields));
		description->read_filters(fields);
		return description;
	}

	/** The paths of the files of the description that load_from_string() makes of @p fields. */
	static QStringList paths_of(pvcop::db::format const& fields)
	{
		QStringList paths;
		for (const std::string& field : fields) {
			if (field.empty()) {
				break;
			}
			paths << QString::fromStdString(field);
		}
		return paths;
	}

	/** Sets the columns and the row filter to those of @p fields, of desc_from_qsetting(). */
	void read_filters(pvcop::db::format const& fields)
	{
		auto field = std::find(fields.begin(), fields.end(), std::string());
		if (field == fields.end() or ++field == fields.end()) {
			return;
		}
		_row_filter = QString::fromStdString(*field);
		_columns.clear();
		while (++field != fields.end()) {
			_columns << QString::fromStdString(*field);
		}
	}

  public:
	void serialize_write(PVCore::PVSerializeObject& so) const override
	{
		so.set_current_status("Saving source file information...");
		so.attribute_write("files_path", _paths);
		so.attribute_write("columns", _columns);
		so.attribute_write("row_filter", _row_filter);
	}

	static std::unique_ptr<PVInputDescription> serialize_read(PVCore::PVSerializeObject& so)
	{
		auto description = std::make_unique<PVFileListDescription>(read_paths(so));
		description->read_filters(so);
		return description;
	}

	/**
	 * Sets the columns and the row filter to those serialize_write() wrote to @p so, which an
	 * investigation saved before them does not hold: it imports all of the rows and columns.
	 */
	void read_filters(PVCore::PVSerializeObject& so)
	{
		_columns = so.attribute_read<QStringList>("columns");
		_row_filter = so.attribute_read<QString>("row_filter");
	}

	/**
	 * The paths of the files of the description that serialize_write() wrote to @p so.
	 *
	 * @throw PVCore::PVSerializeReparaibleFileError if one of the files cannot be read
	 */
	static QStringList read_paths(PVCore::PVSerializeObject& so)
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

		return paths;
	}
};

} // namespace PVRush

#endif // __PVRUSH_PVFILELISTDESCRIPTION_H__
