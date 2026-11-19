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
#include <QString>
#include <QStringList>

namespace PVRush
{

/**
 * An input type whose sources read files one after the other, with a format generated from them:
 * the user chooses the files, and the type describes them and makes their format.
 */
class PVInputTypeFileList : public PVInputTypeDesc<PVFileListDescription>
{
  public:
	bool create_widget(hash_formats& formats,
	                   list_inputs& inputs,
	                   QString& format,
	                   PVCore::PVArgumentList& args_ext,
	                   QWidget* parent = nullptr) const override;

	bool create_widget_with_input_files(const QStringList& input_paths,
	                                    hash_formats& formats,
	                                    list_inputs& inputs,
	                                    QString& format,
	                                    PVCore::PVArgumentList& args_ext,
	                                    QWidget* parent = nullptr) const override;

	/** The files are those of the array "paths" of @p params_json. */
	bool create_source_description_params(const QString& params_json,
	                                      list_inputs& inputs,
	                                      PVFormat& format) const override;

	bool has_generated_format() const override { return true; }

	QString tab_name_of_inputs(list_inputs const& in) const override;

	bool get_custom_formats(PVInputDescription_p /*in*/, hash_formats& /*formats*/) const override
	{
		return false;
	}

	QCursor cursor() const override { return QCursor(Qt::PointingHandCursor); }

  protected:
	/** The title of the dialog that chooses the files. */
	virtual QString files_dialog_title() const = 0;

	/** The filter of the names of the files the dialog shows, such as "Files (*.ext)". */
	virtual QString files_dialog_filter() const = 0;

	/**
	 * Adds the description of the files @p paths to @p inputs, and sets @p format to theirs.
	 *
	 * @return false, after telling the user why, if the files cannot be imported
	 */
	virtual bool
	describe_files(const QStringList& paths, list_inputs& inputs, PVFormat& format) const = 0;
};

} // namespace PVRush

#endif // __PVRUSH_PVINPUTTYPEFILELIST_H__
