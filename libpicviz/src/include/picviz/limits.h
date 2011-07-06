//! \file limits.h
//! $Id: limits.h 2490 2011-04-25 02:07:58Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef _PICVIZ_LIMITS_H_
#define _PICVIZ_LIMITS_H_


#ifdef __cplusplus
 extern "C" {
#endif

/* FIXME: 65535 means unknown */

#ifdef WIN32
#define PICVIZ_MAXFLOAT 65535 
#define PICVIZ_MAXDOUBLE 65535
#define PICVIZ_MINFLOAT -65535 
#define PICVIZ_MINDOUBLE -65535
#else
#define PICVIZ_MAXFLOAT 3.40282347e+38F
#define PICVIZ_MAXDOUBLE 1.79769e+308
#define PICVIZ_MINFLOAT -65535
#define PICVIZ_MINDOUBLE -65535
#endif

#define PICVIZ_IPV4_MAXVAL 4294967295UL /* -> 11111111 11111111 11111111 11111111 */
#define PICVIZ_TIME_24H_MAX 86399
#define PICVIZ_TIME_WEEK_MAX 604793 /* 'Saturday 23:59:59'. Sun = 0, Sat = 6. (6 * 86399 = 518394) */
#define PICVIZ_TIME_MONTH_MAX 2678369
#define PICVIZ_TIME_YEAR_MAX 29030064

#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_AREA_H_ */
