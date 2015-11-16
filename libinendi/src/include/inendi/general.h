/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

/* #ifdef CUSTOMER_RELEASE */
//AUTO_CUSTOMER_RELEASE #include <inendi/api-obfuscate.h> 
/* #endif */

#include <pvbase/types.h>

#include <pvkernel/core/general.h>

/* #define INENDI_BEGIN_NAMESPACE namespace Inendi { */
/* #define INENDI_CLOSE_NAMESPACE } */

#ifndef VM_START
	#define VM_START
	#define VM_END
	#define CODEREPLACE_START
	#define CODEREPLACE_END
#endif

/* Global that holds the error message */
/* extern char *debug_msg;  */

//#ifdef __cplusplus
// extern "C" {
//#endif

#define INENDI_DEFAULT_IMAGE_HEIGHT 350
#define INENDI_DEFAULT_IMAGE_HEADER_HEIGHT 15
#define INENDI_DEFAULT_IMAGE_WIDTH 0

#define INENDI_DEFAULT_PENWIDTH "1.0"

#define INENDI_RING_STRUCTOF(s) _##s
/* #define INENDI_ARRAY_PUSH(array,type) (*((type *)apr_array_push (array))) */
#define INENDI_USERDATA(userdata,type) (*(type *)(userdata))

/* #define INENDI_ARRAY(t) apr_array_header_t * */

#ifdef _POSIX_C_SOURCE
/* /usr/include/python2.6/pyconfig-32.h:1019:1: warning: "_POSIX_C_SOURCE" redefined */
#undef _POSIX_C_SOURCE 
#endif
#ifdef _XOPEN_SOURCE
/* /usr/include/python2.6/pyconfig-32.h:1028:1: warning: "_XOPEN_SOURCE" redefined */
#undef _XOPEN_SOURCE
#endif


#define INENDI_TIME_MSEC    * 1
#define INENDI_TIME_SEC     * 1000 MSEC
#define INENDI_TIME_MIN     * 60 SEC
#define INENDI_TIME_HOUR    * 60 MIN
#define INENDI_TIME_DAY     * 24 HOUR

#define INENDI_OBJECT_DTRI(x) x
#define INENDI_OBJECT_SCENE(x) x
#define INENDI_OBJECT_SOURCE(x) x
#define INENDI_OBJECT_MAPPING(x) x
#define INENDI_OBJECT_MAPPED(x) x
#define INENDI_OBJECT_PLOTTING(x) x
#define INENDI_OBJECT_PLOTTED(x) x
#define INENDI_OBJECT_VIEW(x) x
#define INENDI_OBJECT_SELECTION(x) x
#define INENDI_OBJECT_NRAW(x) x

/* #undef _DEBUG */
/* #include <Python.h> */

/* #include <apr_general.h> */
/* #include <apr_tables.h> */
/* #include <apr_hash.h> */
/* #include <apr_ring.h> */

//#ifdef __cplusplus
// }
//#endif

