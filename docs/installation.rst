.. highlight:: shell

============
Installation
============


Stable release
--------------

To install ``deepcoder``, run this command in your terminal:

.. code-block:: console

    $ python -m pip install deepcoder

This is the preferred method to install ``deepcoder``, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. note::

    When reporting bugs, please specify your installation method (using `pip install` or by building the repository) in the "Installation Method" section of the bug report template.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for ``deepcoder`` can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/deepcode-ai/deepcoder.git

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ cd deepcoder
    $ poetry install
    $ poetry shell


.. _Github repo: https://github.com/deepcode-ai/deepcoder.git

Troubleshooting
---------------

For mac and linux system, there are sometimes slim python installations that do not include the ``deepcoder`` requirement tkinter, which is a standard library and thus not pip installable.

To install tkinter on mac, you can for example use brew:

.. code-block:: console

    $ brew install python-tk

On debian-based linux systems you can use:

.. code-block:: console

    $ sudo apt-get install python3-tk
