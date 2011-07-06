/*! \mainpage PVRush library
 *
 * \section intro Introduction
 *
 * PVRush has the single task to take any type of file (that is stored in a given place)
 * and create a CSV-like version of it that is stored in memory. It also manages the
 * extraction of these data.
 * 
 * This memory CSV-like is called a \b NRAW (for Normalized Raw). 
 *
 * \section How it works
 *
 * 
 * \section Sources in PVRush
 *
 * The interface of a source is defined by \ref PVRawSource. The responsability of a source
 * is to create chunks that are then processed by a TBB pipeline.
 * involved in the creation of a source.
 *
 * \subection Input connectors
 *
 * The input are segmented with two main properties :
 * <ul>
 * <li>where it comes from (a local file, an HDFS cluster, a database). Handled by the "input type" plugins</li>
 * <li>the contained data (PCAP packets, text data, SQL result, etc...). It is handled by the" 
 * </ul>
 *
 * \subsection Aggregator
 *
 * \section Working pipeline
 *
 * \subsection Definition of a filter function
 *
 * \subsection Combining filter functions
 *
 * \subsection Stop conditions of a pipeline
 *
 * \section Memory allocations
 *
 * \section Improvements
 *
 \dot
  digraph picvizarch {
      db [label="database", shape="parallelogram"];
      binary [label="binary file", shape="parallelogram"];
      plaintext [label="plaintext file", shape="parallelogram"];
      format [label="format", shape="invhouse"];
      formatplugins [label="format plugins"];
      formatpluginshelper [label="format plugins helpers"];
      normalize [label="normalize", shape="invhouse"];
      decode [label="decode", shape="invhouse"];
      decodeplugins [label="decode plugins"];
      csv [label="NRAW (csv)", shape="rectangle"];

      db -> format;
      binary -> format;
      plaintext -> format;
      format -> normalize;
      formatplugins -> format;
      format -> formatplugins;
      formatpluginshelper -> formatplugins;
      normalize->decode;
      decode->normalize;      
      decodeplugins -> decode;
      normalize -> csv;
      
  }
 \enddot
 *
 * This is the reference API guide. Choose where to start:
 *
 * - \ref Format Create and use format to cut a file or buffer input in columns
 * - \ref Normalization How to normalize a file into an array
 *
 */

