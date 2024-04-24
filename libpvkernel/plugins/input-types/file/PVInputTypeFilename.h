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

#ifndef SQUEY_PVINPUTTYPEFILENAME_H
#define SQUEY_PVINPUTTYPEFILENAME_H

#include "PVImportFileDialog.h"

#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVFileDescription.h>

#include <QString>
#include <QStringList>
#include <QIcon>
#include <QCursor>

namespace PVRush
{

class PVInputTypeFilename : public PVInputTypeDesc<PVFileDescription>
{
  public:
	PVInputTypeFilename();
	virtual ~PVInputTypeFilename();

  public:
	bool createWidget(hash_formats& formats,
	                  list_inputs& inputs,
	                  QString& format,
	                  PVCore::PVArgumentList& args_ext,
	                  QWidget* parent = nullptr) const override;
	QString name() const override;
	QString human_name() const override;
	QString human_name_serialize() const override;
	QString internal_name() const override;
	QString menu_input_name() const override;
	QString tab_name_of_inputs(list_inputs const& in) const override;
	bool get_custom_formats(PVInputDescription_p in, hash_formats& formats) const override;

	QIcon icon() const override { return QIcon(":/text_icon"); }
	QCursor cursor() const override { return QCursor(Qt::PointingHandCursor); }

  protected:
	/**
	 * Push input corresponding to filenames in "inputs" performing uncompress if required.
	 */
	bool load_files(QStringList const& filenames, list_inputs& inputs, QWidget* parent) const;

  protected:
	mutable QStringList _tmp_dir_to_delete;
	int _limit_nfds;

  protected:
	CLASS_REGISTRABLE_NOCOPY(PVInputTypeFilename)
};
} // namespace PVRush

#endif
