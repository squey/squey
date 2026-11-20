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

#ifndef SQUEY_PVTEXTIMPORTWIDGET_H
#define SQUEY_PVTEXTIMPORTWIDGET_H

#include <pvkernel/widgets/PVImportWidget.h>

class QLineEdit;
class QRadioButton;

namespace PVWidgets
{
class PVImportFilesWidget;
} // namespace PVWidgets

namespace PVRush
{

class PVInputTypeFilename;

/**
 * What the import page asks for text files: the files, whether they make one source or one each,
 * and their format, either one found next to them or else detected from their lines, or one the
 * user chooses.
 */
class PVTextImportWidget : public PVWidgets::PVImportWidget
{
	Q_OBJECT

  public:
	explicit PVTextImportWidget(PVInputTypeFilename const& type, QWidget* parent = nullptr);

  public:
	bool is_ready() const override;
	/**
	 * With an empty format where none was found next to the files, which the import detects then.
	 *
	 * @throw std::runtime_error if a source has more files than can be open at once, or if the
	 * chosen format cannot be read
	 */
	std::vector<source_t> sources() const override;
	void set_paths(QStringList const& paths) override;
	void set_format(QString const& path) override;

  private:
	void choose_format();
	/** That found next to the files of @p inputs, or an empty one to detect. */
	PVFormat format_next_to(QList<PVInputDescription_p> const& inputs) const;

  private:
	PVInputTypeFilename const& _type;
	PVWidgets::PVImportFilesWidget* _files;
	QRadioButton* _detected;
	QRadioButton* _from_file;
	QLineEdit* _format_path;
};

} // namespace PVRush

#endif // SQUEY_PVTEXTIMPORTWIDGET_H
