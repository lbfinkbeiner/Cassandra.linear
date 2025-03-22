This repo is the user-facing code. To see the developer version, consult https://github.com/lbfinkbeiner/cassandra_linear_dev/tree/main

To get started, install this repo in the typical Pythonic fashion (e.g. navigate to directory and run `pip install -e .`). A Cassandra-linear emulator file (.cle) needs to be downloaded as well, but I'm not hosting one at the moment because we need to resolve the crisis in the developer tools.
(I can't remember, but I think the developer tools have to be at least installed for this repo to work...)

Lukas' to-do:
* I need to retire the unc_emu somehow, or at least the use should request it separately. otherwise an underscore must be used (i.e. to indicate that this is an internal fn).
* does the user need to download lil_k? maybe not, since the emu assumes lil_k only in the original predictions--what it returns depends on what k values the user gives

INPUT I STILL NEED TO GET FROM ARIEL AND ANDREA
* How should we distribute the emulator files? Those are extremely heavy and, although serialized, don't make much sense for the repo itself. Can I host the files somewhere else and simply direct the users to them from this README?
* Should we provide various emulator files for various numbers of training samples? e.g. Hnu3_2k Hnu3_3k, Hnu3_4k. Then, we could put a brief table in this README featuring numbers relating the accuracy-versus-speed tradeoff (with more training samples, the emulator file size balloons and the speed of predictions decreases).
