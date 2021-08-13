# MCML




What is mypy?
-------------

Mypy is an optional static type checker for Python.  You can add type
hints ([PEP 484](https://www.python.org/dev/peps/pep-0484/)) to your
Python programs, and use mypy to type check them statically.
Find bugs in your programs without even running them!

You can mix dynamic and static typing in your programs. You can always
fall back to dynamic typing when static typing is not convenient, such
as for legacy code.

Here is a small example to whet your appetite (Python 3):

```python
from typing import Iterator

def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b
```
See [the documentation](https://mypy.readthedocs.io/en/stable/introduction.html) for more examples.

For Python 2.7, the standard annotations are written as comments:
```python
def is_palindrome(s):
    # type: (str) -> bool
    return s == s[::-1]
```

See [the documentation for Python 2 support](https://mypy.readthedocs.io/en/latest/python2.html).

Mypy is in development; some features are missing and there are bugs.
See 'Development status' below.

Requirements
------------

You need Python 3.5 or later to run mypy.  You can have multiple Python
versions (2.x and 3.x) installed on the same system without problems.

In Ubuntu, Mint and Debian you can install Python 3 like this:

    $ sudo apt-get install python3 python3-pip

For other Linux flavors, macOS and Windows, packages are available at

  https://www.python.org/getit/


Quick start
-----------

Mypy can be installed using pip:

    $ python3 -m pip install -U mypy

If you want to run the latest version of the code, you can install from git:

    $ python3 -m pip install -U git+git://github.com/python/mypy.git


Now, if Python on your system is configured properly (else see
"Troubleshooting" below), you can type-check the [statically typed parts] of a
program like this:

    $ mypy PROGRAM

You can always use a Python interpreter to run your statically typed
programs, even if they have type errors:

    $ python3 PROGRAM

You can also try mypy in an [online playground](https://mypy-play.net/) (developed by
Yusuke Miyazaki).

[statically typed parts]: https://mypy.readthedocs.io/en/latest/getting_started.html#function-signatures-and-dynamic-vs-static-typing




### Working with `virtualenv`

If you are using [`virtualenv`](https://virtualenv.pypa.io/en/stable/),
make sure you are running a python3 environment. Installing via `pip3`
in a v2 environment will not configure the environment to run installed
modules from the command line.

    $ python3 -m pip install -U virtualenv
    $ python3 -m virtualenv env



For more on the tests, such as how to write tests and how to control
which tests to run, see [Test README.md](test-data/unit/README.md).




License
-------

Mypy is licensed under the terms of the MIT License (see the file
LICENSE).
