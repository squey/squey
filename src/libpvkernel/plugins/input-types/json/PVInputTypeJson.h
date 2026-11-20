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

#include <memory>

namespace PVRush
{

class PVJsonFileDescription;
class PVJsonSchema;

/**
 * Files of one JSON object per line, such as JSON Lines, whose fields are the axes. Compressed
 * files are decompressed as they are read.
 */
class PVInputTypeJson : public PVInputTypeFileList
{
  public:
	PVInputTypeJson();

  public:
	/** After finding the schema of the files of @p inputs, if they have none. */
	PVFormat get_format_from_inputs(list_inputs const& inputs,
	                                PVCore::PVTaskProgress* progress = nullptr) const override;

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

	QString files_dialog_title() const override;
	QString files_dialog_filter() const override;
	/** With the schema of the files, which reading them all finds. */
	std::shared_ptr<PVFileListDescription> describe(const QStringList& paths,
	                                                PVCore::PVTaskProgress& progress,
	                                                QStringList& warnings) const override;
	std::shared_ptr<PVFileListDescription> with_paths(const PVFileListDescription& description,
	                                                  const QStringList& paths) const override;
	/** From the counts of the values of the schema of the files. */
	QHash<QString, double> fill_rates(const PVFileListDescription& input_desc) const override;
	QString row_filter_error(const PVFileListDescription& input_desc,
	                         const QString& condition) const override;

  private:
	/**
	 * The schema of the files @p paths, while @p progress, if any, shows how far the reading got
	 * and can stop it: nullptr then.
	 *
	 * @param warnings receives, if any, what to tell the user about lines left out
	 * @throw std::runtime_error if a file cannot be read, or if the files have no column
	 */
	static std::shared_ptr<const PVJsonSchema>
	schema_of(const QStringList& paths, PVCore::PVTaskProgress* progress, QStringList* warnings);

  protected:
	CLASS_REGISTRABLE_NOCOPY(PVInputTypeJson)
};

} // namespace PVRush

#endif // __PVINPUTTYPEJSON_H__
