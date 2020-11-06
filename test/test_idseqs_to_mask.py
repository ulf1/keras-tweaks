from keras_tweaks import idseqs_to_mask


def test1():
    x = idseqs_to_mask(2.0)
    assert x == 4.0
