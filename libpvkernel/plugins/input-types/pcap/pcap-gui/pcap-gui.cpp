/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2017
 */

#include "pcap-gui.h"

#include <libpvpcap/ws.h>

#include <QMessageBox>
#include <QFileInfo>

void PVPcapsicum::check_wireshark_profile_exists(rapidjson::Document& json_data)
{
	if (not json_data.IsNull() and json_data.HasMember("options") and
	    json_data["options"].HasMember("wireshark_profile")) {
		const std::string& wireshark_profiles_dir = pvpcap::get_wireshark_profiles_dir();
		const std::string& wireshark_profile_name =
		    json_data["options"]["wireshark_profile"].GetString();
		const std::string& wireshark_profile_path =
		    wireshark_profiles_dir + "/" + wireshark_profile_name;

		if (not QFileInfo(wireshark_profile_path.c_str()).exists()) {
			json_data["options"].RemoveMember("wireshark_profile");
			QMessageBox::warning(
			    nullptr, QString("Wireshark profile '%1' not found")
			                 .arg(QString::fromStdString(wireshark_profile_name)),
			    QString("Wireshark profile '%1' was not found. Falling back to 'Default' profile.")
			        .arg(QString::fromStdString(wireshark_profile_name)),
			    QMessageBox::Ok);
		}
	}
}
