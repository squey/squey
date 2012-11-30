/**
 * \file picviz_stat.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PICVIZSTAT_H
#define PVCORE_PICVIZSTAT_H

#include <iostream>

/**
 * @file picviz_stat.h
 *
 * statistics framework.
 *
 * This module must be used through the following macros: #PV_STAT
 * and derivates.
 */

/**
 * @def PV_STAT(NAME, VALUE)
 *
 * Print a generic statistic line using the parameters.
 *
 * @param NAME the statistic's name
 * @param UNIT the statistic's unit
 * @param VALUE the statistic's value
 */
#define PV_STAT(NAME, UNIT, VALUE) std::cerr << "__pvstat__{{" << (NAME) << "," << (UNIT) << "}}:" << (VALUE) << std::endl


/**
 * @def PV_STAT_MEM_USE(NAME, VALUE)
 *
 * Print a statistic line about memory consumption.
 *
 * @param NAME the statistic's name
 * @param VALUE the statistic's value
 */
#define PV_STAT_MEM_USE(NAME, VALUE) PV_STAT((NAME), "Mio", (VALUE))

/**
 * @def PV_STAT_MEM_BW(NAME, VALUE)
 *
 * Print a statistic line about memory bandwidth.
 *
 * @param NAME the statistic's name
 * @param VALUE the statistic's value
 */
#define PV_STAT_MEM_BW(NAME, VALUE) PV_STAT((NAME), "Mio/s", (VALUE))

/**
 * @def PV_STAT_TIME_SEC(NAME, VALUE)
 *
 * Print a statistic line about time in seconds.
 *
 * @param NAME the statistic's name
 * @param VALUE the statistic's value
 */
#define PV_STAT_TIME_SEC(NAME, VALUE) PV_STAT((NAME), "s", (VALUE))

/**
 * @def PV_STAT_TIME_MSEC(NAME, VALUE)
 *
 * Print a statistic line about time in milliseconds.
 */
#define PV_STAT_TIME_MSEC(NAME, VALUE) PV_STAT((NAME), "ms", (VALUE))

/**
 * @def PV_STAT_TIME_MSEC(NAME, VALUE)
 *
 * Print a statistic line about time in milliseconds.
 */
#define PV_STAT_TIME_MICROSEC(NAME, VALUE) PV_STAT((NAME), "µs", (VALUE))

#endif // PVCORE_PICVIZSTAT_H
