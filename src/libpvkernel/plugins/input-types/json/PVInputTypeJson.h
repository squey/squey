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

#ifndef __PVINPUTTYPEJSON_H__
#define __PVINPUTTYPEJSON_H__

#include <pvkernel/rush/PVInputTypeFileList.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QString>
#include <QStringList>

namespace PVRush
{

class PVJsonFileDescription;

/**
 * Files of one JSON object per line, such as JSON Lines, whose fields are the axes. Compressed
 * files are decompressed as they are read.
 */
class PVInputTypeJson : public PVInputTypeFileList
{
  public:
	PVInputTypeJson();

  public:
	PVFormat get_format_from_inputs(list_inputs const& inputs) const override;

	QString name() const override;
	QString human_name() const override;
	QString human_name_serialize() const override;
	QString internal_name() const override;
	/** Those of JSON files, and the same followed by those of PVJsonLines::compressions(). */
	QStringList get_supported_extensions() const override;
	QString menu_input_name() const override;

	PVModdedIcon icon() const override { return PVModdedIcon("brackets-curly"); }

	PVInputDescription_p serialize_read(PVCore::PVSerializeObject& so) override;
	PVInputDescription_p load_input_from_string(pvcop::db::format const& fields,
	                                            bool multi_inputs) override;

  protected:
	QString files_dialog_title() const override;
	QString files_dialog_filter() const override;
	bool
	describe_files(const QStringList& paths, list_inputs& inputs, PVFormat& format) const override;
	/** From the counts of the values of the schema of the files. */
	QHash<QString, double> fill_rates(const PVFileListDescription& input_desc) const override;
	QString row_filter_error(const PVFileListDescription& input_desc,
	                         const QString& condition) const override;

  private:
	/**
	 * Finds the schema of the files of @p input_desc, while the user sees the progress.
	 *
	 * @return false if the user stopped it, or, after telling the user why, if the files have no
	 * column
	 * @throw std::runtime_error if a file cannot be read
	 */
	bool find_schema(PVJsonFileDescription& input_desc) const;

	/**
	 * Sets @p format to that of the columns that @p input_desc imports, after finding the schema
	 * of its files if it has none.
	 *
	 * @return false as find_schema() does
	 * @throw std::runtime_error as find_schema() does
	 */
	bool find_format(PVJsonFileDescription& input_desc, PVFormat& format) const;

  protected:
	CLASS_REGISTRABLE_NOCOPY(PVInputTypeJson)
};

} // namespace PVRush

#endif // __PVINPUTTYPEJSON_H__
