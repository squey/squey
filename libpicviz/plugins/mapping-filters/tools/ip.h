#ifndef LIBPICVIZ_MAPPING_TOOLS_IP_H
#define LIBPICVIZ_MAPPING_TOOLS_IP_H

#include <stdint.h>
#include <QString>

bool parse_ipv4(QString const& value, uint32_t& intval);

#endif
