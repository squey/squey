/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVVERSION_FILE_H
#define PVVERSION_FILE_H

/* Version definition and macros */
typedef unsigned int version_t;

/*
 * PVCORE_VERSION is (major << 16) + (minor << 8) + patch.
 */
/* #define PVCORE_VERSION 0x010102 for 1.1.2 */
#define INENDI_CURRENT_VERSION_MAJOR ${INSPECTOR_VERSION_MAJOR}
#define INENDI_CURRENT_VERSION_MINOR ${INSPECTOR_VERSION_MINOR}
#define INENDI_CURRENT_VERSION_PATCH ${INSPECTOR_VERSION_MICRO}

#define INENDI_VERSION_NAME "${INSPECTOR_VERSION_NAME}"

#define INENDI_VERSION_INVALID 0

/*
 * Use it like this: if (INENDI_CURRENT_VERSION >= PVCORE_VERSION(1, 1, 2))
 */
#define INENDI_MAJOR_VERSION(v) ((v & 0xFF0000)>>16)
#define INENDI_MINOR_VERSION(v) ((v & 0x00FF00)>>8)
#define INENDI_PATCH_VERSION(v) ((v & 0x0000FF))
#define INENDI_VERSION(major, minor, patch) ((major<<16)|(minor<<8)|(patch))
#define INENDI_VERSION_STR(major, minor, patch) _INENDI_VERSION_STR(major, minor, patch)

#ifdef INENDI_DEVELOPER_MODE
#define _INENDI_VERSION_STR(major, minor, patch) #major "." #minor "." #patch " (dev mode)"
#else
#define _INENDI_VERSION_STR(major, minor, patch) #major "." #minor "." #patch
#endif

#define INENDI_CURRENT_VERSION INENDI_VERSION(INENDI_CURRENT_VERSION_MAJOR, INENDI_CURRENT_VERSION_MINOR, INENDI_CURRENT_VERSION_PATCH)
#define INENDI_CURRENT_VERSION_STR INENDI_VERSION_STR(INENDI_CURRENT_VERSION_MAJOR, INENDI_CURRENT_VERSION_MINOR, INENDI_CURRENT_VERSION_PATCH)

/* PVConfig settings name */
#define PVCONFIG_LAST_KNOWN_CUR_RELEASE "version/last_known_cur_release"
#define PVCONFIG_LAST_KNOWN_MAJ_RELEASE "version/last_known_maj_release"

#endif
