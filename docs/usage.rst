How to use the hame module
==========================
In a typical usage, you will call a routine of the hame library to compute a matrix element.

Some examples are given below. A Python script containing these examples is available :doc:`here <script>`.

.. _usage:


One-photon matrix element
-------------------------
To compute numerically the dipole matrix element between the 1S and 2P states of the hydrogen atom, you may use:

>>> import hame  
>>> n, l = 1, 0
>>> nprime, lprime = 2, 1
>>> nsup = 20
>>> print('<',n,l,'| z |',nprime,lprime,'> from Gordon formula    :',hame.gordon_formula(n, l, nprime, lprime))
< 1 0 | z | 2 1 > from Gordon formula    : 0.7449355390278027
>>> print('<',n,l,'| z |',nprime,lprime,'> from numerics          :',hame.compute_dipole_matrix_element(n, l, nprime, lprime, nsup))
< 1 0 | z | 2 1 > from numerics          : 0.7449355390278033

Two-photon matrix element
-------------------------

Light-shift
-----------

