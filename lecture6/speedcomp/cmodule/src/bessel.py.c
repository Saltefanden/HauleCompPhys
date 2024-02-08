#include <stdint.h>
#include <stdio.h>
#include <bessel.py.h>
#include <bessel.h>

PyObject *bessel2d(PyObject *self, PyObject *args){
  PyArrayObject *inarr;
  int N;
  double ulim;
  int l;
  PyArg_ParseTuple(args, "Oidi", &inarr, &N, &ulim, &l);
  if (PyErr_Occurred()) {
    printf("You cocked it at PyArg_ParseTuple\n");
    return NULL;
  }

  
  int64_t size = PyArray_SIZE(inarr);
  double *data;
  PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
  int nd = 1;
  npy_intp dims[] = {[0] = size};
  PyArray_AsCArray((PyObject **) &inarr, &data, dims, nd, descr );
  if (PyErr_Occurred()) {
    printf("Things happened with the array conversion\n");
    return NULL;
  }
  besselup_2d(data, N, ulim, l);
  return (PyObject *) inarr;
}

static PyMethodDef bessel_meths[] = {
  {"besselc", bessel2d, METH_VARARGS, "A fucking fast bessel function from c\n"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef filenamemodule = {
  PyModuleDef_HEAD_INIT,
  "besselcc",
  "Module documentation todo\n",
  -1,
  bessel_meths,
};

PyMODINIT_FUNC PyInit_besselcc(){
  PyObject *module = PyModule_Create(&filenamemodule);
  import_array();
  return module;
}

