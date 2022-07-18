=======
flowgym
=======


.. image:: https://img.shields.io/pypi/v/flowgym.svg
        :target: https://pypi.python.org/pypi/flowgym

.. image:: https://img.shields.io/travis/SiggyF/flowgym.svg
        :target: https://travis-ci.com/SiggyF/flowgym

.. image:: https://readthedocs.org/projects/flowgym/badge/?version=latest
        :target: https://flowgym.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




OpenAI Gym environment for navigating through flow fields.


* Free software: GNU General Public License v3
* Documentation: https://flowgym.readthedocs.io.


Features
--------

You can register the environment by importing the flowgym package:

```
import flowgym
```

After that the gym environment is available through `gym.make`.

```
# a gym environment with a velocity field
env = gym.make('flowgym/FlowWorldEnv-v0')

# gym environment with only source and target
env = gym.make('flowgym/WorldEnv-v0')

```

See the notebooks directory for a (non-working) example of how to use the gym in tf-agents.




Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
