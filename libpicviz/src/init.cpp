//! \file init.cpp
//! $Id: init.cpp 2520 2011-04-30 12:26:48Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/general.h>
#include <picviz/init.h>

// #include <apr_general.h>

#include <pvcore/environ.h>

#include <stdio.h>
#include <stdlib.h>

/**
 * \ingroup PicvizMain
 * @{
 */

/** \file init.c
 * \brief Library initialization
 */


/*! \mainpage Picviz library
 *
 * \section intro Introduction
 *
 * Picviz library provides API for programmers to use
 * picviz features. The pcv program is the biggest user
 * of the picviz library, you may have a look at it.
 *
 * From the very beginning, Picviz was designed to be embeded
 * by third party software. This is why this library.
 * 
 * The Picviz library allows to access to the parser, the
 * rendering and their plugins, but also to output plugins.
 *
 * Picviz has the following architecture:
 \dot
  digraph picvizarch {
node [shape=record];
      format [label="Format (syslog, pcap)"];
      source [label="Source"];
      mapped [label="Mapped"];
      plotted [label="Plotted"];
      view [label="View"];
      dotssource [label="..."];
      dotsmapped [label="..."];
      dotsplotted [label="..."];
      
      format -> source;
      source -> mapped [label="Mapping function"];
      source -> dotssource;
      mapped -> plotted [label="Plotting function"];
      mapped -> dotsmapped;
      plotted -> view;
      plotted -> dotsplotted;

  }
 \enddot
 *
 * This is the reference API guide. Choose where to start:
 *
 * - \ref PicvizFormat Write format plugins and descriptions
 *
 * - Write plugins: functions, normalize and parsers
 *     -# \ref PicvizFunctions Create new mapping and plotting functions
 *     -# \ref PicvizNormalize Tell Picviz how to open a file
 *     -# \ref PicvizParsers Use PCRE to normalize a text file
 *
 * - \ref PicvizLayers Understand how layers work
 *
 * - \ref PicvizFilters Write filtering plugins
 *
 * - \ref PicvizMappingVariables How to graph an extracted string onto an axis
 *
 * - \ref PicvizPlottingVariables How to graph an extracted string onto an axis
 *
 */


/**
 * Initialize the Picviz library
 *
 * @param argc: Address of the argc parameter of your main() function.
 * @param argv: Address of the argv parameter of your main() function.
 *
 * @return NULL if there is an error, the library context otherwise.
 */
void picviz_init(int argc, char **argv)
{
	// apr_status_t status;
	// /* picviz_context_t *context = NULL; */

	// // PVCore::mkenv("LC_ALL", "C");



	// status = apr_initialize();
	// if ( status != APR_SUCCESS ) {
	// 	fprintf(stderr, "Error initializing APR library!\n");
	// 	return;
	// }

	// /* context = picviz_context_new(); */
	// /* if ( ! context ) { */
	// /* 	fprintf(stderr, "Cannot create a new context!\n"); */
	/* 	return NULL; */
	/* } */

	/* return context; */
}

/**
 * Terminates the Picviz library
 */
void picviz_terminate(void)
{
	// apr_terminate();
	/* free(context); */
}

/*@}*/
