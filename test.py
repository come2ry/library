import sys


def imp_tests(test_path_list):
    # import importlib
    import importlib.util
    g = globals()
    for test_path, imp_name in test_path_list:
        spec = importlib.util.spec_from_file_location(imp_name, test_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        g[imp_name] = module


imp_tests([
    ("test/verify/aoj_1549.test.py", "aoj_1549")
])
aoj_1549.main()
