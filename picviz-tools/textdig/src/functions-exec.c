#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <picviz/file.h>

#include <textdig/textdig.h>

picviz_file_t *file = NULL;

static char *last_var_name = NULL;
static char *var_obj_type = NULL;

void print_args(char **args)
{
	int i;

	for (i = 0; args[i] != NULL; i++) {
		printf("%d:%s\n", i, args[i]);
	}

}

void handle_file_object(char *member)
{
  if (!strcmp(member, "lines")) {
    printf("%ld\n", file->nblines);
  }
}


void *function_run_from_string(const char *string)
{
	char *str_copied;
	size_t str_len;	

	char **args;
	char *command;
	char *retvar = NULL;
	char *arg;
	char *member;
	int i;

	str_copied = strdup(string);
	str_len = strlen(str_copied);
	str_copied[str_len-1] = '\0';

	/* Read args */
	i = 0;
	arg = strtok(str_copied, " ");
	args = malloc(sizeof(char *));
	while (arg) {
		args[i] = strdup(arg);
		arg = strtok(NULL, " ");
		i++;
		args = realloc(args, (i+1) * sizeof(char *));
	}
	args[i] = NULL;

	print_args(args);


	/* We deal with a command that should return a variable */
	if ((args[1]) && (!strcmp(args[1], "="))) {
		retvar = args[0];
		command = args[2];
	}

	command = args[0];

	printf("last var name = %s\n", last_var_name);

	if (last_var_name) {
		if (!strncmp(last_var_name, args[0], strlen(last_var_name))) {
			member = strtok(args[0], ".");
			if (member) {
				member = strtok(NULL, ".");

				if (!strcmp(var_obj_type, "file")) {
					handle_file_object(member);
				}

			} else {
				printf("Error with variable '%s'\n", args[0]);
				return NULL;
			}
		}

		/* last_var_name = NULL; */
	}

	/* Run core functions */
	if (!strcmp("print", command)) {
	  printf("textdig v%d.%d\n", TEXTDIG_VER_MAJOR, TEXTDIG_VER_MINOR);
	}

	if ((!strcmp("quit", command)) || 
	    (!strcmp("exit", command))) {
	    exit(0);
	}

	/* functions dealing with files */
	if (!strcmp("open", command)) {
		if (!retvar) {
			printf("error: no variable provided!\n");
			printf("var = open filename\n");
			return NULL;
		}
		/* ret = open filename.log */
		printf("AAA\n");
		file = picviz_file_new(args[3]);
		var_obj_type = "file";
		last_var_name = strdup(args[0]);
		return NULL;
	}

	if (!strcmp("close", command)) {
		picviz_file_destroy(file);
		return NULL;
	}

	var_obj_type = NULL;

}
