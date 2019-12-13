Testing
=======


We use the pytest framework to run the test cases. For example,


```console
$ pytest
============================= test session starts ==============================
platform linux -- Python 3.6.8, pytest-5.2.2, py-1.8.0, pluggy-0.13.0
rootdir: /raid/weilinxu/coder/spr_secure_intelligence-trusted_federated_learning
collected 2 items

test_example.py .F                                                       [100%]

=================================== FAILURES ===================================
_____________________________ test_example_failed ______________________________

    def test_example_failed():
>       assert 5 != 5
E       assert 5 != 5

test_example.py:5: AssertionError
========================= 1 failed, 1 passed in 0.54s ==========================
```