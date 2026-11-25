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

#ifndef __PVWIDGETS_PVFILELISTIMPORTWIDGET_H__
#define __PVWIDGETS_PVFILELISTIMPORTWIDGET_H__

#include <pvkernel/widgets/PVImportWidget.h>

#include <QStringList>

#include <memory>
#include <vector>

namespace PVRush
{
class PVFileListDescription;
class PVInputTypeFileList;
} // namespace PVRush

namespace PVWidgets
{

class PVImportFilesWidget;
class PVImportFilterWidget;

/**
 * What the import page asks for files that an input type reads one after the other: the files,
 * whether they make one source or one each, and which of their columns and rows to import, once
 * reading them found their columns.
 */
class PVFileListImportWidget : public PVImportWidget
{
	Q_OBJECT

  public:
	explicit PVFileListImportWidget(const PVRush::PVInputTypeFileList& type,
	                                QWidget* parent = nullptr);

  public:
	bool is_ready() const override;
	std::vector<source_t> sources() const override;
	void set_paths(QStringList const& paths) override;
	bool takes_files() const override { return true; }
	void add_paths(QStringList const& paths) override;

  private:
	/** Reads the files, as a task of the page, for their columns. */
	void describe();
	/**
	 * A description of the files @p paths among those read, with the columns and the rows
	 * chosen.
	 */
	std::shared_ptr<PVRush::PVFileListDescription> description_of(QStringList const& paths) const;

  private:
	const PVRush::PVInputTypeFileList& _type;
	/** Of all of the files, once read. */
	std::shared_ptr<PVRush::PVFileListDescription> _description;
	PVImportFilesWidget* _files;
	PVImportFilterWidget* _filter;
};

} // namespace PVWidgets

#endif // __PVWIDGETS_PVFILELISTIMPORTWIDGET_H__
