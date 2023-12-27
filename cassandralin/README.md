# Readme

# Next steps:
# get an uncertainty emulator for the massless-neutrino emu.

>># get growth factor from brenda lib
unfortunately, it doesn't look like this is going to install with
    Windows. We could purloin Matteo's code, but then that makes
    code maintenance more difficult going forward.
    
    Windows version of brenda_lib please??

# expand this as a repo where you plug in parameters and get
    # a power spectrum
# write paper
# no wiggle, de-wiggle, and dimensionless power spectrum

# Should we re-do the Hnu2 unc emu with a set of 5000 samples?
# I used 7000 for the test run and now the emu object is huge...
# In fact, maybe all of the emu's would benefit from smaller sizes,
# since we already seem to be in the regime of diminishing returns.

# We also need to figure out how to broaden these priors...
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