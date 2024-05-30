/* * MIT License
 *
 * © ESI Group, 2015
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

#ifndef PVRUSH_PVIMPORTFILEDIALOG_H
#define PVRUSH_PVIMPORTFILEDIALOG_H

#include <pvkernel/widgets/PVFileDialog.h>

class QComboBox;

namespace PVRush
{

/**
 * \class PVImportFileDialog
 *
 * Speciale FileDialog with option to choose the format.
 */
class PVImportFileDialog : public PVWidgets::PVFileDialog
{
	Q_OBJECT

  public:
	PVImportFileDialog(QStringList pluginslist, QWidget* parent = 0);

	/**
	 * Open the FileDialog and return list of selected file with selected format.
	 */
	QStringList getFileNames(QString& treat_as);

  private:
	QComboBox* treat_as_combobox; //!< Combo box with formats.
};
}

#endif // PVIMPORTFILEDIALOG_H
