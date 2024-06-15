To get started, install this repo as well as this other repo (LINK) in the typical Pythonic fashion (e.g. navigate to directory and run `pip install -e .`) and download "Hnu4c_wiggler.cle" from the following Google Drive link:

##NEEDS##

To predict the power spectrum for a particular cosmology,

To-do before we can publish:

* We need to retire the unc_emu somehow, or at least the use should request it separately. otherwise an underscore must be used

* does the user need to download lil_k? maybe not, since the emu assumes lil_k only in the original predictions--what it returns
depends on what k values the user gives


####



How to use this?

Instructions for this package will come at a later time.

You'll need build tools in case you're trying to wire Brenda on
Windows: https://visualstudio.microsoft.com/visual-cpp-build-tools/
But, in the end, it probably won't work anyway. I wasn't ever able to install
Brenda on Windows...

INPUT I STILL NEED TO GET FROM ARIEL AND ANDREA
* How should we distribute the emulator files? Those are extremely heavy and, although serialized, don't make much sense for the repo itself. Can I host the files somewhere else and simply direct the users to them from this README?
* What if I made various emulator files for various numbers of training samples? e.g. Hnu3_2k Hnu3_3k, Hnu3_4k. Then, we could put a brief table in this README featuring numbers relating the accuracy-versus-speed tradeoff.