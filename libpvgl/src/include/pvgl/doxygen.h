/*! \mainpage PVKernel library
 *
 * The PVGL library handles all the OpenGL part of the application. 
 * 
 * \section pvglarch Architecture
 \dot
digraph G {
compound=true;
   subgraph cluster1 {
   label="PVGL";
     subgraph cluster1 {
     label="WTK";

      node1 [label="freeglut3"];
      node2 [label="QT"];
      node3 [label="..."];
     }
     node4 [label="PVAxes"];
     node4 [label="PVWidgetManager"];
   }
}
 \enddot
 * \section wtkintro WTK: Window Toolkit
 *
 * WTK is a wrapper so the OpenGL code has no direct interaction with the window toolkit used
 * to display elements. 
 *
 */

