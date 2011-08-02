//! \file arguments.cpp
//! $Id: arguments.cpp 2489 2011-04-25 01:53:05Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include <pvkernel/core/general.h>

#include <picviz/arguments.h>


picviz_arguments_t *picviz_arguments_new(void)
{
        picviz_arguments_t *arguments;
	arguments = (picviz_arguments_t *)malloc(sizeof(picviz_arguments_t));
	if (!arguments) {
		PVLOG_ERROR("Cannot allocate memory for arguments!\n");
		return NULL;
	}
	arguments->args = NULL;
	arguments->nelts = 0;

	return arguments;
}

void picviz_arguments_destroy(picviz_arguments_t *arguments)
{
	free(arguments);
  
}

int picviz_arguments_item_append(picviz_arguments_t *arguments, picviz_argument_item_t item)
{
	if (!arguments->nelts) {
		arguments->args = (picviz_argument_item_t *)malloc(sizeof(picviz_argument_item_t));
	} else {
		arguments->args = (picviz_argument_item_t *)realloc(arguments->args, (arguments->nelts + 1) * sizeof(picviz_argument_item_t));
	}
	arguments->args[arguments->nelts] = item;
	arguments->nelts++;

	return 0;
}

int picviz_arguments_item_list_append(picviz_arguments_t *arguments, picviz_argument_item_t items[])
{

	int i;

	for (i = 0; i < PICVIZ_ARGUMENTS_MAX; i++) {
		if (items[i].type == PICVIZ_ARGUMENT_TYPE_NONE) break;

		picviz_arguments_item_append(arguments, items[i]);
	}

	return 0;
}

picviz_argument_item_t picviz_arguments_get_item_from_group_and_dualslider_position(picviz_arguments_t *arguments, char *group, picviz_argument_slider_position_t position)
{
	int i;

	if (arguments) {
		for (i=0; i<arguments->nelts; i++) {
			if (!strcmp(arguments->args[i].group, group)) {
				if (arguments->args[i].slider_position == position) {
					return arguments->args[i];
				}
			}
		}
	}

	PVLOG_ERROR("(%s) Cannot get argument from group '%s' and position '%d'!\n", __FUNCTION__, group, position);
}

picviz_argument_item_t picviz_arguments_get_item_from_name(picviz_arguments_t *arguments, const char *name)
{
	int i;

	if (arguments) {
		for (i=0; i<arguments->nelts; i++) {
			if (!strcmp(arguments->args[i].name, name)) {
				return arguments->args[i];
			}
		}
	}

	PVLOG_ERROR("(%s) Cannot get argument named '%s'!\n", __FUNCTION__, name);
}

void picviz_arguments_set_item_from_name(picviz_arguments_t *arguments, char *name, picviz_argument_item_t item)
{
	int i;

	if (arguments) {
		for (i=0; i<arguments->nelts; i++) {
			if (!strcmp(arguments->args[i].name, name)) {
				arguments->args[i] = item;
				return;
			}
		}
	}

	PVLOG_ERROR("Cannot set argument named '%s'!\n", name);
}

char *picviz_arguments_item_get_string(picviz_argument_item_t item)
{
  return item.strval;
}

picviz_argument_item_t picviz_arguments_item_set_string(picviz_argument_item_t item, char *value)
{
  PVLOG_DEBUG("%s: set item with value '%s'\n", __FUNCTION__, value);
  item.strval = strdup(value);
  return item;
}

float picviz_arguments_item_get_float(picviz_argument_item_t item)
{
  return item.fval;
}

float picviz_arguments_read_item_value_as_float(picviz_argument_item_t item)
{
  PVLOG_ERROR("%s: function depracted\n", __FUNCTION__);

  return item.fval;
}

int picviz_arguments_check(picviz_arguments_t *arguments)
{
	return 1;
}

void picviz_arguments_item_debug(picviz_argument_item_t item)
{
  switch(item.type) {
		case PICVIZ_ARGUMENT_STRING:
		  printf("string: '%s'\n", item.strval);
		  break;
		case PICVIZ_ARGUMENT_INT:
		  printf("int: '%d'\n", item.ival);
		  break;
		case PICVIZ_ARGUMENT_FLOAT:
		  printf("float: '%f'\n", item.fval);
		  break;
		case PICVIZ_ARGUMENT_DOUBLE:
		  printf("double: '%f'\n", item.dval);
		  break;
		default:
		  break;
  }
}

void picviz_arguments_debug(picviz_arguments_t *arguments)
{
        int i;

	for (i=0; i<arguments->nelts; i++) {
		printf("Argument '%d' name '%s'\n", i, arguments->args[i].name);

		picviz_arguments_item_debug(arguments->args[i]);
	}
	
}

picviz_argument_item_t picviz_arguments_item_new(void)
{
        picviz_argument_item_t item;

	item.elem = NULL;
	item.strval = NULL;
	item.fval = 0;
	item.ival = 0;
	item.dval = 0;
	item.name = NULL;
	item.type = PICVIZ_ARGUMENT_TYPE_NONE;
	item.widget = PICVIZ_ARGUMENT_WIDGET_NONE;
	item.slider_position = PICVIZ_ARGUMENT_SLIDER_POSITION_NONE;
	item.limit_min = 0;
	item.limit_max = 0;
	item.increase_factor = 0.1;
	
	return item;
}

void picviz_arguments_item_destroy(picviz_argument_item_t item)
{
  if (item.strval) {
    free(item.strval);
  }
}
