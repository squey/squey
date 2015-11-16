/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <apr_general.h>
#include <apr_dso.h>
#include <apr_hash.h>
#include <apr_file_info.h>
#include <apr_strings.h>

#include <inendi/general.h>
#include <inendi/debug.h>
#include <inendi/filters.h>
#include <inendi/utils.h>
#include <inendi/datatreerootitem.h>
#include <inendi/filters.h>
#include <inendi/plugins.h>

/******************************************************************************
 *
 * inendi_filter_plugin_load
 *
 *****************************************************************************/
inendi_filter_t *inendi_filter_plugin_load(apr_pool_t *pool, char *filepath)
{
	apr_dso_handle_t *dso = NULL;
	char errbuf[1024];
	apr_dso_handle_sym_t ressym;
	apr_status_t status;

	inendi_filter_t *filter;

	VM_START  

	filter = (inendi_filter_t *)apr_palloc(pool, sizeof(inendi_filter_t));
	if (!filter) {
		inendi_debug(INENDI_DEBUG_CRITICAL, "Cannot allocate inendi_filter_t in %s:%s\n", __FILE__, __FUNCTION__);
		return NULL;
	}

	inendi_debug(INENDI_DEBUG_NOTICE, "Loading filter plugin '%s'\n", filepath);

	status = apr_dso_load(&filter->dso, filepath, pool);

	status = apr_dso_sym(&ressym, filter->dso, inendi_filtering_function_init_string);
	if (status) {
		apr_dso_error(filter->dso, errbuf, sizeof(errbuf));
		inendi_debug(INENDI_DEBUG_CRITICAL, "Cannot get inendi_filtering_function_init symbol from '%s': %s\n", filepath, errbuf);
		apr_dso_unload(filter->dso);
		return NULL;
	}
	filter->init_func = (inendi_filtering_function_init_func)ressym;

	status = apr_dso_sym(&ressym, filter->dso, inendi_filtering_function_get_type_string);
	if (status) {
		apr_dso_error(filter->dso, errbuf, sizeof(errbuf));
		inendi_debug(INENDI_DEBUG_CRITICAL, "Cannot get inendi_filtering_function_get_type symbol from '%s': %s\n", filepath, errbuf);
		apr_dso_unload(filter->dso);
		return NULL;
	}
	filter->get_type_func = (inendi_filtering_function_get_type_func)ressym;

	status = apr_dso_sym(&ressym, filter->dso, inendi_filtering_function_get_arguments_string);
	if (status) {
		apr_dso_error(filter->dso, errbuf, sizeof(errbuf));
		inendi_debug(INENDI_DEBUG_CRITICAL, "Cannot get inendi_filtering_function_get_arguments symbol from '%s': %s\n", filepath, errbuf);
		apr_dso_unload(filter->dso);
		return NULL;
	}
	filter->get_arguments_func = (inendi_filtering_function_get_arguments_func)ressym;

	status = apr_dso_sym(&ressym, filter->dso, inendi_filtering_function_exec_string);
	if (status) {
		apr_dso_error(filter->dso, errbuf, sizeof(errbuf));
		inendi_debug(INENDI_DEBUG_CRITICAL, "Cannot get inendi_filtering_function_exec symbol from '%s': %s\n", filepath, errbuf);
		apr_dso_unload(filter->dso);
		return NULL;
	}
	filter->exec_func = (inendi_filtering_function_exec_func)ressym;

	status = apr_dso_sym(&ressym, filter->dso, inendi_filtering_function_terminate_string);
	if (status) {
		apr_dso_error(filter->dso, errbuf, sizeof(errbuf));
		inendi_debug(INENDI_DEBUG_CRITICAL, "Cannot get inendi_filtering_function_terminate symbol from '%s': %s\n", filepath, errbuf);
		apr_dso_unload(filter->dso);
		return NULL;
	}
	filter->terminate_func = (inendi_filtering_function_terminate_func)ressym;	

	VM_END

	return filter;
}



/******************************************************************************
 *
 * inendi_filters_plugin_register_all
 *
 *****************************************************************************/
int inendi_filters_plugin_register_all(apr_pool_t *pool, apr_hash_t *hash)
{
	char *pluginsdir;
	apr_dir_t *dir;
	apr_finfo_t dirent;
	apr_status_t status;
	char *ext;
	char *filepath;
	char *extracted_name;

	inendi_filter_t *filter;

	VM_START


	pluginsdir = inendi_plugins_get_filters_dir();

	/* printf("Functions plugin dir:'%s'\n", pluginsdir); */

	status = apr_dir_open(&dir, pluginsdir, pool);
	if (status != APR_SUCCESS) {
		inendi_debug(INENDI_DEBUG_CRITICAL, "Cannot open directory '%s'! Unable to load filters plugins.\n", pluginsdir);
		return FALSE;
	}

	while ((apr_dir_read(&dirent, APR_FINFO_NAME, dir)) == APR_SUCCESS) {
		ext = (char *)strrchr(dirent.name, '.');
		if (!ext) continue;
		if (!strcmp(ext, INENDI_DLL_EXTENSION)) {
			filepath = apr_pstrcat(pool, pluginsdir, INENDI_PATH_SEPARATOR, dirent.name, NULL);
			extracted_name = inendi_utils_plugin_extract_name(filepath);
			filter = inendi_filter_plugin_load(pool, filepath);
			if (!filter) {
				inendi_debug(INENDI_DEBUG_CRITICAL, "Cannot register functions for plugin '%s'\n", filepath);
				continue;
			} else {
				filter->init_func();
				if (hash) {
					extracted_name = strchr(extracted_name, '_');
					extracted_name++;
					apr_hash_set(hash, extracted_name, APR_HASH_KEY_STRING, filter);
				} else {
					inendi_debug(INENDI_DEBUG_CRITICAL, "The filters hash is empty!\n");
					return FALSE;
				}
			}
		}
	}

	apr_dir_close(dir);

	VM_END

	return TRUE;
}



/******************************************************************************
 *
 * inendi_filters_foreach_filter
 *
 *****************************************************************************/
void inendi_filters_foreach_filter(inendi_datatreerootitem_t *datatree, inendi_filtering_function_foreach_func foreach_func, void *userdata)
{
	apr_hash_index_t *hi;

	for (hi = apr_hash_first(NULL, datatree->filters_plugin); hi; hi = apr_hash_next(hi)) {
		const char *key;
		char *extracted_name;
		inendi_filter_t *filter;

		apr_hash_this(hi, (const void**)&key, NULL, (void**)&filter);
		foreach_func((char *)key, filter, userdata);
		//printf("ht iteration: key=%s, val=%X\n", key, filter);
	}

}



/******************************************************************************
 *
 * inendi_filters_get_filter_from_name
 *
 *****************************************************************************/
inendi_filter_t *inendi_filters_get_filter_from_name(inendi_datatreerootitem_t *datatree, char *name)
{
	return (inendi_filter_t *)apr_hash_get(datatree->filters_plugin, name, strlen(name));
}
