/**
 * \file console.c
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <histedit.h> 
#include <termcap.h>

#include <signal.h>


#include <textdig/textdig.h>
#include <textdig/functions-exec.h>

char *prompt(EditLine *e) {
  return "pic> ";
}


void leave_signal(int signum)
{
  exit(0);
}

int interactive_console_start(textdig_options_t options, int argc, char **argv) 
{
	EditLine *el;
	History *myhistory;

	int count;
	const char *line;
	int keepreading = 1;
	HistEvent ev;

	el = el_init(argv[0], stdin, stdout, stderr);
	el_set(el, EL_PROMPT, &prompt);
	el_set(el, EL_EDITOR, "emacs");

	signal(SIGTERM, leave_signal);

	myhistory = history_init();
	if (myhistory == 0) {
		fprintf(stderr, "history could not be initialized\n");
    		return 1;
  	}

  	history(myhistory, &ev, H_SETSIZE, 800);

  	el_set(el, EL_HIST, history, myhistory);

  	while (keepreading) {
    		line = el_gets(el, &count);
		if (!line) {
			printf("\n");
			leave_signal(-1);
		}
    	/* In order to use our history we have to explicitly add commands
    	to the history */
    		if (count > 0) {
      			history(myhistory, &ev, H_ENTER, line);
			function_run_from_string(line);
	      		/* printf("You typed \"%s\"", line); */
    		}
  	}
  

  	/* Clean up our memory */
  	history_end(myhistory);
  	el_end(el);

  	return 0;
}


