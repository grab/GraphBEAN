# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file


def dict_addto(res: dict, a: dict) -> dict:
    for k in res.keys():
        res[k] += float(a[k])
    return res


def dict_div(res: dict, div) -> dict:
    for k in res.keys():
        res[k] /= div
    return res
