//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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
