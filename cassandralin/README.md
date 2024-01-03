emulator_interface takes a little while to import since it loads
all three emulators into main memory...

This is a memory-intensive application!
Make sure that you have at least 2.6 GB of free memory available!

# Readme

# Next steps:
1. Complete all docstrings
2. Get PEP conformity
3. Add unit tests until 100% code coverage is achieved.

# get an uncertainty emulator for the massless-neutrino emu.

>># get growth factor from brenda lib
unfortunately, it doesn't look like this is going to install with
    Windows. We could purloin Matteo's code, but then that makes
    code maintenance more difficult going forward.
    
    Windows version of brenda_lib please??

A. write paper
B. no wiggle, de-wiggle, and dimensionless power spectrum
C. Expand documentation of the development code!!

# Maybe all of the emu's would benefit from smaller sizes,
# since we already seem to be in the regime of diminishing returns.
# That would benefit users who don't want to download 2 GB of emulators...

# We also need to figure out how to broaden these priors even further...
# Mess around with CLASS?
    # On ice until I can get back to a Linux machine.
    # Technically I have a VM on this laptop, but I don't want to
    # split the compute so unfavorably.
    
    # I just tried on Schuschnigg, and I have to say, this still doesn't seem
    # right. The installation spits out errors even though I can technically
    # import classy afterward...
    
# 27 Dec 2023 @ 12:18 we've just discovered that the maximum k in CLASS
# is roughly 1.1 / Mpc. That's kind of bad, but mostly fine since we can't
# probe that small anyway... 

# 27 Dec 2023 @ 12:33 I've spent some time fiddling around but I cannot get
# CLASS to accept negative redshifts. There may be a way, but I haven't found
# it yet...

# @ 13:30 I just went back and checked CAMB again. This time, it's giving me a
# segfault instead of a complaint that the redshift is negative. A segfault is
# significantly more difficult to address...

# sigma12 non-linear sampling is probably a waste of time, but
# we should come back to it in the future.