import json

import numpy as np
from offloader import Offloader, OffloadVector

url = "offload.dt4si.nl"
offloader = Offloader(url, "api/v1", offload_folder="tmp")


def pre(task_folder, x):
    output_data = {"X": x}
    with open(task_folder / "input.json", "w") as outp:
        json.dump(output_data, outp)


def post(task_folder, x):
    with open(task_folder / "output.json", "r") as inp:
        input_f = json.load(inp)

    return input_f["y"]


def hartmann_offload(X):
    # url = "localhost:8080"
    task_resources = {"requests": {"memory": "100Mi", "cpu": "200m"}}

    vector = []
    for x in X:
        vector.append({"x": x.tolist()})
    command = "ls && pip install numpy && python3 offload_test_functions.py"
    off = OffloadVector(
        offloader,
        pre,
        post,
        command,
        "python:3",
        vector,
        task_resources=task_resources,
        local=False,
        auto_delete=False,
        stable=False,
    )
    off.add_file(
        "tests" "\\integration_tests\\offload_test_functions.py",
        des_path="",
    )
    off.get_file("output.json")

    res = []
    for i in off.run():
        res += i

    return np.array(res)
