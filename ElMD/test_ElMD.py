"""Basic tests taken from docs (assuming that docs are correct)."""
from numpy.testing import assert_almost_equal, assert_allclose

from ElMD import ElMD

def test_ElMD():
    x = ElMD("CaTiO3")
    d = x.elmd("SrTiO3")
    assert_almost_equal(d, 0.2)


def test_atomic():
    x = ElMD("CaTiO3", metric="atomic")
    d = x.elmd("SrTiO3")
    assert_almost_equal(d, 3.6)


def test_magpie_sc():
    elmd = ElMD(metric="magpie_sc").elmd
    d = elmd("NaCl", "LiCl")
    assert_almost_equal(d, 0.688539)


def test_magpie():
    x = ElMD("NaCl", metric="magpie")
    d = x.elmd("LiCl")
    assert_almost_equal(d, 46.697806)


def test_featurizingDict():
    featurizingDict = ElMD(metric="magpie").periodic_tab
    check = [
        2.0,
        22.98976928,
        370.87,
        1.0,
        3.0,
        166.0,
        0.93,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        29.2433333333,
        0.0,
        0.0,
        229.0,
    ]
    assert_allclose(featurizingDict["Na"], check)


if __name__ == "__main__":
    test_ElMD()
    test_atomic()
    test_magpie_sc()
    test_magpie()
    test_featurizingDict()
