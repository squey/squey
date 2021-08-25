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

#ifndef FILECONNECTIONDIALOG_H
#define FILECONNECTIONDIALOG_H

#include "logviewer_export.h"
#include "connectionsettings.h"

#include <QDialog>

#include <pvkernel/widgets/PVFileDialog.h>

class QLineEdit;
class QPushButton;

class FileNameSelectorWidget : public QWidget
{
	Q_OBJECT
  public:
	explicit FileNameSelectorWidget(QWidget* parent = 0);
	~FileNameSelectorWidget();
	QString text() const;
	void setText(const QString&);

  private Q_SLOTS:
	void slotPathChanged();

  private:
	QLineEdit* m_path;
	QPushButton* m_selectPath;
	PVWidgets::PVFileDialog _file_dlg;
};

class LOGVIEWER_EXPORT FileConnectionDialog : public QDialog
{
	Q_OBJECT
  public:
	explicit FileConnectionDialog(QWidget* parent);
	~FileConnectionDialog();

	RegisteredFile registeredFileSettings() const;

	void initialize(const RegisteredFile& registered, const QString& hostname);
  private Q_SLOTS:
	void slotTextChanged(const QString& text);
	void slotProtocolChanged(int index);

  private:
	class FileConnectionDialogPrivate;
	FileConnectionDialogPrivate* d;
};

#endif /* FILECONNECTIONDIALOG_H */
