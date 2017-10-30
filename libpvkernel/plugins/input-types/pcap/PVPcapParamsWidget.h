/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2017
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

  public:
	pvpcap::splitted_files_t csv_paths() const { return _selection_widget->csv_paths(); }
	QStringList pcap_paths() const { return _selection_widget->pcap_paths(); }
	QDomDocument get_format() const { return _selection_widget->get_format(); }
	bool is_canceled() { return _selection_widget->is_canceled(); }

  private:
	void open_inspector();

  private:
	SelectionWidget* _selection_widget; //!< Widget with selection information
};

} // namespace PVPcapsicum

#endif // __PVPCAPPARAMSWIDGET_H__
