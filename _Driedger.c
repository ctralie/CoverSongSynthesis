/*Programmer: Chris Tralie
*Purpose: Code for doing Driedger updates in place
*Helps from following links:
*http://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html
*https://scipy-lectures.github.io/advanced/interfacing_with_c/interfacing_with_c.html
*Special thanks especially to http://dan.iel.fm/posts/python-c-extensions/*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include "Driedger.h"

/* Docstrings */
static char module_docstring[] =
    "This module provides an implementation of Driedger's update rules on an H matrix";
static char driedgerupdates_docstring[] =
    "Perform Driedger's update rules on an H matrix";
static char diagupdates_docstring[] =
    "Perform Driedger's diagonal update rules on an H matrix";

/* Available functions */
static PyObject* Driedger_DriedgerUpdates(PyObject* self, PyObject* args);
static PyObject* Driedger_DiagUpdates(PyObject* self, PyObject* args);

/* Module specification */
static PyMethodDef module_methods[] = {
    {"DriedgerUpdates", Driedger_DriedgerUpdates, METH_VARARGS, driedgerupdates_docstring},
    {"DiagUpdates", Driedger_DiagUpdates, METH_VARARGS, diagupdates_docstring},
    {NULL, NULL, 0, NULL}
};



#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef Driedger =
{
    PyModuleDef_HEAD_INIT,
    "_Driedger", /* name of module */
    module_docstring, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    module_methods
};

/* Initialize the module */
PyMODINIT_FUNC PyInit__Driedger(void)
{
    /* Load `numpy` functionality. */
    import_array();
    return PyModule_Create(&Driedger);
}
#else
/* Initialize the module */
PyMODINIT_FUNC init_Driedger(void)
{
    PyObject *m = Py_InitModule3("_Driedger", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}
#endif


static PyObject *Driedger_DriedgerUpdates(PyObject *self, PyObject *args)
{
    PyObject *H_obj;
    int r, p, c;
    float iterfac;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "Oiiif", &H_obj, &r, &p, &c, &iterfac))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *H_array = PyArray_FROM_OTF(H_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    
    /* If that didn't work, throw an exception. */
    if (H_array == NULL) {
        Py_XDECREF(H_array);
        return NULL;
    }

    int M = (int)PyArray_DIM(H_array, 0);
    int N = (int)PyArray_DIM(H_array, 1);

    /* Get pointers to the data as C-types. */
    double *H = (double*)PyArray_DATA(H_array);

    /* H updates */
    DriedgerUpdates(H, M, N, r, p, c, iterfac);    

    /* Clean up. */
    Py_DECREF(H_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", 0.0);
    return ret;
}


static PyObject *Driedger_DiagUpdates(PyObject *self, PyObject *args)
{
    PyObject *H_obj;
    int c;
    float iterfac;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "Oif", &H_obj, &c, &iterfac))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *H_array = PyArray_FROM_OTF(H_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    
    /* If that didn't work, throw an exception. */
    if (H_array == NULL) {
        Py_XDECREF(H_array);
        return NULL;
    }

    int M = (int)PyArray_DIM(H_array, 0);
    int N = (int)PyArray_DIM(H_array, 1);

    /* Get pointers to the data as C-types. */
    double *H = (double*)PyArray_DATA(H_array);

    /* H updates */
    DiagUpdates(H, M, N, c, iterfac);    

    /* Clean up. */
    Py_DECREF(H_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", 0.0);
    return ret;
}