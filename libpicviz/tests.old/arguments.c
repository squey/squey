#include <stdio.h>

#include <picviz/arguments.h>

int main(int argc, char **argv)
{
  picviz_arguments_t *arguments;
  picviz_argument_item_t item;
  picviz_argument_item_t items[] = {
    PICVIZ_ARGUMENTS_STRING_GROUP("Name of this argument", "my group", "mystring")
    PICVIZ_ARGUMENTS_TEXTBOX(Search, NULL, "")
    PICVIZ_ARGUMENTS_INT("My integer", 1234)
    PICVIZ_ARGUMENTS_END
};
  int i;

  arguments = picviz_arguments_new();

  picviz_arguments_item_list_append(arguments, items);
  item = picviz_arguments_item_new();

  item.type = PICVIZ_ARGUMENT_STRING;
  item.name = "this other one that I have!";
  item.strval = "This is quite a nice string!!!";
  picviz_arguments_item_append(arguments, item);
  
  picviz_arguments_debug(arguments);

  item = picviz_arguments_get_item_from_name(arguments, "this other one that I have!");
  if (strcmp(item.strval, "This is quite a nice string!!!")) {
    printf("The element was not properly saved!\n");	
    return 1;
  }

  picviz_arguments_destroy(arguments);

  arguments = picviz_arguments_new();

  picviz_arguments_item_list_append(arguments, items);
  item = picviz_arguments_get_item_from_name(arguments, "Search");
  item = picviz_arguments_item_set_string(item, "Something I want to keep");
  picviz_arguments_set_item_from_name(arguments, item.name, item);

  picviz_arguments_debug(arguments);

  picviz_arguments_destroy(arguments);

  return 0;
}
