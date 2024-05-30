/**
 * @file
 *
 */

#ifndef PVVERSION_FILE_H
#define PVVERSION_FILE_H

/* Version definition and macros */
typedef unsigned int version_t;

/*
 * PVCORE_VERSION is (major << 16) + (minor << 8) + patch.
 */
/* #define PVCORE_VERSION 0x010102 for 1.1.2 */
#define SQUEY_CURRENT_VERSION_MAJOR ${SQUEY_VERSION_MAJOR}
#define SQUEY_CURRENT_VERSION_MINOR ${SQUEY_VERSION_MINOR}
#define SQUEY_CURRENT_VERSION_PATCH ${SQUEY_VERSION_MICRO}

#define SQUEY_VERSION_NAME "${SQUEY_VERSION_NAME}"

#define SQUEY_VERSION_INVALID 0

/*
 * Use it like this: if (SQUEY_CURRENT_VERSION >= PVCORE_VERSION(1, 1, 2))
 */
#define SQUEY_VERSION(major, minor, patch) ((major<<16)|(minor<<8)|(patch))
#define SQUEY_VERSION_STR(major, minor, patch) _SQUEY_VERSION_STR(major, minor, patch)

#ifdef SQUEY_DEVELOPER_MODE
#define _SQUEY_VERSION_STR(major, minor, patch) #major "." #minor "." #patch ".dev"
#else
#define _SQUEY_VERSION_STR(major, minor, patch) #major "." #minor "." #patch
#endif

#define SQUEY_CURRENT_VERSION SQUEY_VERSION(SQUEY_CURRENT_VERSION_MAJOR, SQUEY_CURRENT_VERSION_MINOR, SQUEY_CURRENT_VERSION_PATCH)
#define SQUEY_CURRENT_VERSION_STR SQUEY_VERSION_STR(SQUEY_CURRENT_VERSION_MAJOR, SQUEY_CURRENT_VERSION_MINOR, SQUEY_CURRENT_VERSION_PATCH)

#endif
