/**
 * \file time.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef _PICVIZ_TIME_H_
#define _PICVIZ_TIME_H_

#ifdef __cplusplus
 extern "C" {
#endif



/******************************************************************************
 *
 * TYPEDEF
 *
 *****************************************************************************/

typedef struct _picviz_time_t picviz_time_t;



/******************************************************************************
 *
 * STRUCTURE
 *
 *****************************************************************************/

struct _picviz_time_t {
	/* based on struct tm and extended */
	int sec;         /* seconds */
	int min;         /* minutes */
	int hour;        /* hours */
	int mday;        /* day of the month */
	int mon;         /* month */
	int year;        /* year */
	int wday;        /* day of the week */
	int yday;        /* day in the year */
	int isdst;       /* daylight saving time */

	int gmt;	    /* gmt zone. 1 for Paris, -5 for Mexico .. */
	long usec;	    /* micro seconds */
};

#ifdef __cplusplus
 }
#endif

#endif /* _PICVIZ_TIME_H_ */
