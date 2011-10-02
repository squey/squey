#ifndef PVVERSION_FILE_H
#define PVVERSION_FILE_H

/* Version definition and macros */
typedef unsigned int version_t;

/*
 * PVCORE_VERSION is (major << 16) + (minor << 8) + patch.
 */
/* #define PVCORE_VERSION 0x010102 for 1.1.2 */
#define PICVIZ_CURRENT_VERSION_MAJOR 2
#define PICVIZ_CURRENT_VERSION_MINOR 1
#define PICVIZ_CURRENT_VERSION_PATCH 1

#define PICVIZ_VERSION_INVALID 0

/*
 * Use it like this: if (PICVIZ_CURRENT_VERSION >= PVCORE_VERSION(1, 1, 2))
 */
#define PICVIZ_MAJOR_VERSION(v) ((v & 0xFF0000)>>16)
#define PICVIZ_MINOR_VERSION(v) ((v & 0x00FF00)>>8)
#define PICVIZ_PATCH_VERSION(v) ((v & 0x0000FF))
#define PICVIZ_VERSION(major, minor, patch) ((major<<16)|(minor<<8)|(patch))
#define PICVIZ_VERSION_STR(major, minor, patch) _PICVIZ_VERSION_STR(major, minor, patch)
#define _PICVIZ_VERSION_STR(major, minor, patch) #major "." #minor "." #patch

#define PICVIZ_CURRENT_VERSION PICVIZ_VERSION(PICVIZ_CURRENT_VERSION_MAJOR, PICVIZ_CURRENT_VERSION_MINOR, PICVIZ_CURRENT_VERSION_PATCH)
#define PICVIZ_CURRENT_VERSION_STR PICVIZ_VERSION_STR(PICVIZ_CURRENT_VERSION_MAJOR, PICVIZ_CURRENT_VERSION_MINOR, PICVIZ_CURRENT_VERSION_PATCH)

/* PVConfig settings name */
#define PVCONFIG_LAST_KNOWN_CUR_RELEASE "version/last_known_cur_release"
#define PVCONFIG_LAST_KNOWN_MAJ_RELEASE "version/last_known_maj_release"

#endif
