#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ── helpers ──────────────────────────────────────────────────────────────── */

static double safe_float(PyObject *v) {
    if (!v || v == Py_None) return 0.0;
    if (PyBool_Check(v))   return PyObject_IsTrue(v) ? 1.0 : 0.0;
    if (PyLong_Check(v))   { double r = PyLong_AsDouble(v);  return isfinite(r) ? r : 0.0; }
    if (PyFloat_Check(v))  { double r = PyFloat_AsDouble(v); return isfinite(r) ? r : 0.0; }
    return 0.0;
}

static double dict_f(PyObject *d, const char *k) {
    if (!d || !PyDict_Check(d)) return 0.0;
    PyObject *v = PyDict_GetItemString(d, k);
    return v ? safe_float(v) : 0.0;
}

static double nested_f(PyObject *d, const char *outer, const char *inner) {
    if (!d || !PyDict_Check(d)) return 0.0;
    PyObject *o = PyDict_GetItemString(d, outer);
    if (!o || !PyDict_Check(o)) return 0.0;
    PyObject *v = PyDict_GetItemString(o, inner);
    return v ? safe_float(v) : 0.0;
}

/* ── get_obs ──────────────────────────────────────────────────────────────────
 * Fixed-layout observation vector. Every slot has a known meaning so the
 * policy network can actually learn. Total: 40 floats.
 * All values normalised to roughly [-1, 1] or [0, 1] range.
 * --------------------------------------------------------------------------- */

#define OBS_SIZE 40

static PyObject *get_obs(PyObject *self, PyObject *args) {
    PyObject *q;          /* qualities dict */
    Py_ssize_t dummy;     /* ignored state_size arg kept for API compat */
    if (!PyArg_ParseTuple(args, "On", &q, &dummy)) return NULL;

    npy_intp dims[1] = {OBS_SIZE};
    PyObject *arr = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (!arr) return NULL;
    float *b = (float *)PyArray_DATA((PyArrayObject *)arr);
    memset(b, 0, OBS_SIZE * sizeof(float));

    if (!q || q == Py_None || !PyDict_Check(q)) return arr;

    /* Party vote shares — normalise by /50 so ~[0,1] */
    b[ 0] = (float)(dict_f(q, "spd_votes")   / 50.0);
    b[ 1] = (float)(dict_f(q, "nsdap_votes") / 50.0);
    b[ 2] = (float)(dict_f(q, "kpd_votes")   / 50.0);
    b[ 3] = (float)(dict_f(q, "z_votes")     / 50.0);
    b[ 4] = (float)(dict_f(q, "ddp_votes")   / 50.0);
    b[ 5] = (float)(dict_f(q, "dvp_votes")   / 50.0);
    b[ 6] = (float)(dict_f(q, "dnvp_votes")  / 50.0);
    b[ 7] = (float)(dict_f(q, "other_votes") / 50.0);

    /* Relations — typically -100..100, normalise by /100 */
    b[ 8] = (float)(dict_f(q, "kpd_relation")  / 100.0);
    b[ 9] = (float)(dict_f(q, "z_relation")    / 100.0);
    b[10] = (float)(dict_f(q, "ddp_relation")  / 100.0);
    b[11] = (float)(dict_f(q, "dvp_relation")  / 100.0);
    b[12] = (float)(dict_f(q, "dnvp_relation") / 100.0);
    b[13] = (float)(dict_f(q, "lvp_relation")  / 100.0);
    b[14] = (float)(dict_f(q, "nsdap_relation")/ 100.0);

    /* Hindenburg */
    b[15] = (float)(dict_f(q, "hindenburg_relation") / 100.0);
    b[16] = (float)(dict_f(q, "hindenburg_angry"));   /* bool 0/1 */

    /* Government health */
    b[17] = (float)(dict_f(q, "coalition_dissent") / 100.0);
    b[18] = (float)(dict_f(q, "dissent_percent")   / 100.0);
    b[19] = (float)(dict_f(q, "pro_republic")      / 100.0);

    /* Resources */
    b[20] = (float)(dict_f(q, "resources") / 20.0);   /* usually 0-20 */
    b[21] = (float)(dict_f(q, "budget")    / 100.0);

    /* Time — encode as fraction through the game (1928=0, 1934=1) */
    double year  = dict_f(q, "year");
    double month = dict_f(q, "month");
    double t = ((year - 1928.0) * 12.0 + month) / 72.0; /* 6 years = 72 months */
    b[22] = (float)fmax(0.0, fmin(1.0, t));

    /* Reichsbanner / paramilitary */
    b[23] = (float)(dict_f(q, "rb_strength")   / 100.0);
    b[24] = (float)(dict_f(q, "rb_military")   / 100.0);

    /* Advisors & actions remaining */
    b[25] = (float)(dict_f(q, "n_advisors")    / 5.0);
    b[26] = (float)(dict_f(q, "month_actions") / 5.0);

    /* Government position flags */
    b[27] = (float)(dict_f(q, "in_government"));   /* bool */
    b[28] = (float)(dict_f(q, "is_chancellor"));   /* bool */
    b[29] = (float)(dict_f(q, "in_coalition"));    /* bool */
    b[30] = (float)(dict_f(q, "rubicon"));         /* crossed rubicon flag */
    b[31] = (float)(dict_f(q, "schleicher_path")); /* DNEF path flag */

    /* Landtag/Reichstag seats — normalise by /300 */
    b[32] = (float)(dict_f(q, "spd_seats")   / 300.0);
    b[33] = (float)(dict_f(q, "nsdap_seats") / 300.0);

    /* Economics */
    b[34] = (float)(dict_f(q, "unemployment") / 30.0);  /* % unemployment */
    b[35] = (float)(dict_f(q, "economic_growth") / 10.0);

    /* KPD faction state (conciliators etc) */
    b[36] = (float)(dict_f(q, "kpd_conciliators")); /* bool */
    b[37] = (float)(dict_f(q, "kpd_left_faction") / 100.0);

    /* spare slots for future use */
    b[38] = 0.0f;
    b[39] = 0.0f;

    return arr;
}

/* ── compute_reward ───────────────────────────────────────────────────────────
 * Delta-based reward. Returns change in a weighted sum of game metrics.
 * Terminal rewards (+100 win, -100 lose) are handled in env.py.
 * --------------------------------------------------------------------------- */

static PyObject *compute_reward(PyObject *self, PyObject *args) {
    PyObject *prev, *cur;
    if (!PyArg_ParseTuple(args, "OO", &prev, &cur)) return NULL;

    double r = 0.0;

    /* ── deltas (current - previous, scaled) ── */

    /* SPD vote share: most important, big weight */
    r += 1.5 * (nested_f(cur,  "ai_party_support", "SPD")
              - nested_f(prev, "ai_party_support", "SPD"));

    /* Nazi vote share: penalise growth heavily */
    r -= 2.0 * (nested_f(cur,  "ai_party_support", "NSDAP")
              - nested_f(prev, "ai_party_support", "NSDAP"));

    /* Pro-republic sentiment */
    r += 0.8 * (dict_f(cur, "pro_republic") - dict_f(prev, "pro_republic"));

    /* Dissent: penalise increases */
    r -= 0.6 * (dict_f(cur, "dissent_percent") - dict_f(prev, "dissent_percent"));

    /* Hindenburg: reward improving relation, penalise making him angry */
    r += 0.4 * (dict_f(cur,  "hindenburg_relation")
              - dict_f(prev, "hindenburg_relation"));
    /* Only penalise the *transition* to angry, not every step while angry */
    double hind_delta = dict_f(cur, "hindenburg_angry") - dict_f(prev, "hindenburg_angry");
    if (hind_delta > 0.0) r -= 4.0;   /* just became angry */
    if (hind_delta < 0.0) r += 2.0;   /* calmed down */

    /* Coalition dissent */
    r -= 0.5 * (dict_f(cur, "coalition_dissent") - dict_f(prev, "coalition_dissent"));

    /* Party relations */
    const char *rel_parties[] = {"Z", "DDP", "DVP", "KPD", "LVP"};
    for (int i = 0; i < 5; i++) {
        r += 0.15 * (nested_f(cur,  "relations", rel_parties[i])
                   - nested_f(prev, "relations", rel_parties[i]));
    }
    /* Penalise getting too cosy with NSDAP */
    r -= 0.2 * (nested_f(cur,  "relations", "NSDAP")
              - nested_f(prev, "relations", "NSDAP"));

    /* Resources: small reward for gaining, small penalty for spending */
    r += 0.1 * (dict_f(cur, "resources") - dict_f(prev, "resources"));

    /* Reichstag/Landtag seats — big reward for gains */
    double seat_delta = (dict_f(cur,  "spd_seats") - dict_f(prev, "spd_seats"))
                      - (dict_f(cur,  "nsdap_seats") - dict_f(prev, "nsdap_seats"));
    r += 0.5 * seat_delta;

    /* Rubicon: reaching it isn't a loss but it's bad — penalise crossing it */
    double rubicon_delta = dict_f(cur, "rubicon") - dict_f(prev, "rubicon");
    if (rubicon_delta > 0.0) r -= 5.0;

    /* ── small per-step bonuses for sustained good states ── */
    /* These are small so they don't dominate the delta signals */
    if (nested_f(cur, "ai_party_support", "NSDAP") < 20.0) r += 0.05;
    if (nested_f(cur, "ai_party_support", "SPD")   > 30.0) r += 0.05;
    if (dict_f(cur, "pro_republic")  > 55.0) r += 0.04;
    if (dict_f(cur, "dissent_percent") < 15.0) r += 0.04;
    if (dict_f(cur, "hindenburg_angry") > 0.5) r -= 0.1; /* sustained anger tax */

    /* clamp to prevent exploding gradients */
    if (r >  20.0) r =  20.0;
    if (r < -20.0) r = -20.0;

    return PyFloat_FromDouble(r);
}

/* ── module boilerplate ───────────────────────────────────────────────────── */

static PyMethodDef methods[] = {
    {"get_obs",        get_obs,        METH_VARARGS, "Fixed-layout observation vector from qualities."},
    {"compute_reward", compute_reward, METH_VARARGS, "Delta-based RL reward."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "fast_env", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_fast_env(void) {
    import_array();
    return PyModule_Create(&module);
}