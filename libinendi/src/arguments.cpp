/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include <pvkernel/core/general.h>

#include <inendi/arguments.h>


inendi_arguments_t *inendi_arguments_new(void)
{
        inendi_arguments_t *arguments;
	arguments = (inendi_arguments_t *)malloc(sizeof(inendi_arguments_t));
	if (!arguments) {
		PVLOG_ERROR("Cannot allocate memory for arguments!\n");
		return NULL;
	}
	arguments->args = NULL;
	arguments->nelts = 0;

	return arguments;
}

void inendi_arguments_destroy(inendi_arguments_t *arguments)
{
	free(arguments);
  
}

int inendi_arguments_item_append(inendi_arguments_t *arguments, inendi_argument_item_t item)
{
	if (!arguments->nelts) {
		arguments->args = (inendi_argument_item_t *)malloc(sizeof(inendi_argument_item_t));
	} else {
		arguments->args = (inendi_argument_item_t *)realloc(arguments->args, (arguments->nelts + 1) * sizeof(inendi_argument_item_t));
	}
	arguments->args[arguments->nelts] = item;
	arguments->nelts++;

	return 0;
}

int inendi_arguments_item_list_append(inendi_arguments_t *arguments, inendi_argument_item_t items[])
{

	int i;

	for (i = 0; i < INENDI_ARGUMENTS_MAX; i++) {
		if (items[i].type == INENDI_ARGUMENT_TYPE_NONE) break;

		inendi_arguments_item_append(arguments, items[i]);
	}

	return 0;
}

inendi_argument_item_t inendi_arguments_get_item_from_group_and_dualslider_position(inendi_arguments_t *arguments, char *group, inendi_argument_slider_position_t position)
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

inendi_argument_item_t inendi_arguments_get_item_from_name(inendi_arguments_t *arguments, const char *name)
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

void inendi_arguments_set_item_from_name(inendi_arguments_t *arguments, char *name, inendi_argument_item_t item)
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

char *inendi_arguments_item_get_string(inendi_argument_item_t item)
{
  return item.strval;
}

inendi_argument_item_t inendi_arguments_item_set_string(inendi_argument_item_t item, char *value)
{
  PVLOG_DEBUG("%s: set item with value '%s'\n", __FUNCTION__, value);
  item.strval = strdup(value);
  return item;
}

float inendi_arguments_item_get_float(inendi_argument_item_t item)
{
  return item.fval;
}

float inendi_arguments_read_item_value_as_float(inendi_argument_item_t item)
{
  PVLOG_ERROR("%s: function depracted\n", __FUNCTION__);

  return item.fval;
}

int inendi_arguments_check(inendi_arguments_t *arguments)
{
	return 1;
}

void inendi_arguments_item_debug(inendi_argument_item_t item)
{
  switch(item.type) {
		case INENDI_ARGUMENT_STRING:
		  printf("string: '%s'\n", item.strval);
		  break;
		case INENDI_ARGUMENT_INT:
		  printf("int: '%d'\n", item.ival);
		  break;
		case INENDI_ARGUMENT_FLOAT:
		  printf("float: '%f'\n", item.fval);
		  break;
		case INENDI_ARGUMENT_DOUBLE:
		  printf("double: '%f'\n", item.dval);
		  break;
		default:
		  break;
  }
}

void inendi_arguments_debug(inendi_arguments_t *arguments)
{
        int i;

	for (i=0; i<arguments->nelts; i++) {
		printf("Argument '%d' name '%s'\n", i, arguments->args[i].name);

		inendi_arguments_item_debug(arguments->args[i]);
	}
	
}

inendi_argument_item_t inendi_arguments_item_new(void)
{
        inendi_argument_item_t item;

	item.elem = NULL;
	item.strval = NULL;
	item.fval = 0;
	item.ival = 0;
	item.dval = 0;
	item.name = NULL;
	item.type = INENDI_ARGUMENT_TYPE_NONE;
	item.widget = INENDI_ARGUMENT_WIDGET_NONE;
	item.slider_position = INENDI_ARGUMENT_SLIDER_POSITION_NONE;
	item.limit_min = 0;
	item.limit_max = 0;
	item.increase_factor = 0.1;
	
	return item;
}

void inendi_arguments_item_destroy(inendi_argument_item_t item)
{
  if (item.strval) {
    free(item.strval);
  }
}
