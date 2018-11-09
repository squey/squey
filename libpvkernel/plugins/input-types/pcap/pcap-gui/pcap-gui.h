/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2017
 */

#ifndef INENDI_PCAPGUI_H
#define INENDI_PCAPGUI_H

#include <rapidjson/document.h>

namespace PVPcapsicum
{

void check_wireshark_profile_exists(rapidjson::Document& json_data);
}

#endif /* INENDI_PCAPGUI_H */
