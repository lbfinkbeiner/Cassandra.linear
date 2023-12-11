from cassL import camb_interface as ci

# Without these keys, a dictionary cannot qualify as a cosmology.
essential_keys = ["h", "ombh2", "omch2", "OmK", "omnuh2", "A_s", "n_s", "w0",
                  "wa"]

def test_scale_sigma12():
    """
    Make sure that the function scale_sigma12 is working correctly.
    """
    # Make sure that 'conversions' matches the input cosmology:
    raise NotImplementedError
