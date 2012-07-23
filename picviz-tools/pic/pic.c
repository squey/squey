/**
 * \file pic.c
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <histedit.h> 
#include <termcap.h>

char *prompt(EditLine *e) {
  return "pic> ";
}

int main(int argc, char **argv) 
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

	myhistory = history_init();
	if (myhistory == 0) {
		fprintf(stderr, "history could not be initialized\n");
    		return 1;
  	}

  	history(myhistory, &ev, H_SETSIZE, 800);

  	el_set(el, EL_HIST, history, myhistory);

  	while (keepreading) {
    		line = el_gets(el, &count);

    	/* In order to use our history we have to explicitly add commands
    	to the history */
    		if (count > 0) {
      			history(myhistory, &ev, H_ENTER, line);
	      		printf("You typed \"%s\"", line);
    		}
  	}
  

  	/* Clean up our memory */
  	history_end(myhistory);
  	el_end(el);

  	return 0;
}


