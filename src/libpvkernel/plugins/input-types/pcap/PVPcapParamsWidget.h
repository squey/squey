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

#ifndef __PVPCAPPARAMSWIDGET_H__
#define __PVPCAPPARAMSWIDGET_H__

#include <QDialog>
#include <QLayout>
#include <QString>
#include <QStringList>

#include "pcap/pcap-gui/import-ui/src/include/SelectionWidget.h"

namespace PVPcapsicum
{

class PVInputTypePcap;

class PVPcapParamsWidget : public QDialog
{
	Q_OBJECT

  public:
	PVPcapParamsWidget(QWidget* parent);
	PVPcapParamsWidget(const QStringList& pcap_paths, QWidget* parent);

  public:
	pvpcap::splitted_files_t csv_paths() const { return _selection_widget->csv_paths(); }
	QStringList pcap_paths() const { return _selection_widget->pcap_paths(); }
	QDomDocument get_format() const { return _selection_widget->get_format(); }
	bool is_canceled() { return _selection_widget->is_canceled(); }

  private:
	void open_squey();

  private:
	SelectionWidget* _selection_widget; //!< Widget with selection information
};

} // namespace PVPcapsicum

#endif // __PVPCAPPARAMSWIDGET_H__
