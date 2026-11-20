/* * MIT License
 *
 * © Squey, 2026
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

#ifndef __PVRUSH_PVINPUTTYPEFILELIST_H__
#define __PVRUSH_PVINPUTTYPEFILELIST_H__

#include <pvkernel/rush/PVFileListDescription.h>
#include <pvkernel/rush/PVInputType.h>

#include <QCursor>
#include <QHash>
#include <QString>
#include <QStringList>

#include <memory>

namespace PVRush
{

/**
 * An input type whose sources read files one after the other, with a format generated from them:
 * the user chooses the files, then which of their columns and rows to import, in the import page
 * of the main window, and the type describes them and makes their format.
 */
class PVInputTypeFileList : public PVInputTypeDesc<PVFileListDescription>
{
  public:
	PVWidgets::PVImportWidget* create_import_widget(QWidget* parent) const override;

	/**
	 * The files are those of the array "paths" of @p params_json. Its array "columns" and its
	 * string "row_filter" choose the columns and the rows to import, if it has them.
	 */
	bool create_source_description_params(const QString& params_json,
	                                      list_inputs& inputs,
	                                      PVFormat& format,
	                                      PVCore::PVTaskProgress& progress) const override;

	bool has_generated_format() const override { return true; }

	QString tab_name_of_inputs(list_inputs const& in) const override;

	bool get_custom_formats(PVInputDescription_p /*in*/, hash_formats& /*formats*/) const override
	{
		return false;
	}

	QCursor cursor() const override { return QCursor(Qt::PointingHandCursor); }

  public:
	/** The title of the dialog that chooses the files. */
	virtual QString files_dialog_title() const = 0;

	/** The filter of the names of the files the dialog shows, such as "Files (*.ext)". */
	virtual QString files_dialog_filter() const = 0;

	/**
	 * The description of the files @p paths, out of the GUI thread: it may take long to read
	 * them, such as the JSON files whose schema it finds, while @p progress shows how far it got
	 * and can stop it.
	 *
	 * @param warnings receives what to tell the user, such as lines left out
	 * @return nullptr if @p progress was canceled
	 * @throw std::runtime_error with what to tell the user, if the files cannot be imported
	 */
	virtual std::shared_ptr<PVFileListDescription>
	describe(const QStringList& paths, PVCore::PVTaskProgress& progress, QStringList& warnings) const = 0;

	/**
	 * A description of the files @p paths, among those of @p description, with what reading
	 * them found, such as their schema, but none of the columns and rows it chose.
	 */
	virtual std::shared_ptr<PVFileListDescription>
	with_paths(const PVFileListDescription& description, const QStringList& paths) const = 0;

	/**
	 * The share of the rows of the files of @p input_desc that give a column a value, from 0 to 1,
	 * by the names of the columns of which the type knows it.
	 */
	virtual QHash<QString, double> fill_rates(const PVFileListDescription& /*input_desc*/) const
	{
		return {};
	}

	/**
	 * What is wrong with @p condition, as a condition on the rows of the files of @p input_desc,
	 * or nothing.
	 */
	virtual QString row_filter_error(const PVFileListDescription& input_desc,
	                                 const QString& condition) const = 0;
};

} // namespace PVRush

#endif // __PVRUSH_PVINPUTTYPEFILELIST_H__
