emulator_interface takes a little while to import since it loads
all three emulators into main memory...

This is a memory-intensive application!
Make sure that you have at least 2.6 GB of free memory available!

# Next steps in the development of this project:
0. Procure some speed tests for the paper.
0. New priors for the massless-neutrino emulator.
0. Test the whole pipeline from start to finish, preferably with only COMET
    priors to get rid of extreme cosmologies.
1. no wiggle and de-wiggle
2. write paper
3. Also obtain general feedback about the script, announce it to the LSS group.
4. Add unit tests until 100% code coverage is achieved.
5. A lot of the docstrings contain repeated verbose chunks, e.g. the
    description of a fully filled-in Brenda cosmology. What is the recommended
    way of doing this?
6. Expand documentation of the development code!!

Unfortunately, it doesn't look like the full Brendalib package will install on
with Windows. We could purloin just the script in which we are interested
(cosmo_tools.py), but then that makes code maintenance more difficult going
forward. Windows version of brenda_lib please??

Other thoughts:
* Maybe all of the emu's would benefit from smaller sizes (e.g., N_s=3000)
    since we already seem to be in the regime of diminishing returns. That
    would benefit users who don't want to download 2 GB of emulators...
* sigma12 non-linear sampling is probably a waste of time, but
    we should come back to it in the future.


# We would also like to figure out how to broaden these priors even further...
* Mess around with CLASS? I just tried on Schuschnigg, and I have to say, this
    still doesn't seem right. The installation spits out errors even though I
    can technically import classy afterward...
    
* 27 Dec 2023 @ 12:18 discovered that the maximum k in CLASS is roughly
    1.1 / Mpc. That's kind of bad, but mostly fine since we can't probe that
    small anyway...

* 27 Dec 2023 @ 12:33 I've spent some time fiddling around but I cannot get
    CLASS to accept negative redshifts. There may be a way, but I haven't found
    it yet...

* @ 13:30 I just went back and checked CAMB again. This time, it's giving me a
    segfault instead of a complaint that the redshift is negative. A segfault
    is a significantly more difficult to address...

