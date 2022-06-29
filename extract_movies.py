# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

with open(f"data/movies.txt", "r", errors="ignore") as infile:
    movies = infile.readlines()

infile.close()
with open(f"data/movies.csv", "w") as out:
    out.write(
        "product_id,user_id,profile_name,helpfulness_numerator,helpfulness_denominator,score,time,summary,text\n"
    )

    count = 0
    out_dict = {
        "product/productId": "",
        "review/userId": "",
        "review/profileName": "",
        "review/helpfulness": "",
        "review/score": "",
        "review/time": "",
        "review/summary": "",
        "review/text": "",
        "hnum": 0,
        "hden": 0,
    }

    for row in movies:
        if row.rstrip() != "":
            cells = row.split(":")
            if cells[0] == "product/productId":
                if len(cells) > 1:
                    out_dict[cells[0]] = (
                        cells[1]
                        .replace(",", "")
                        .replace("\n", "")
                        .replace("<br />", "")
                        .replace("\\", "")
                        .strip()
                    )
                if count > 0:
                    output = (
                        f"{out_dict['product/productId']},{out_dict['review/userId']},{out_dict['review/profileName']},{out_dict['hnum']},{out_dict['hden']},"
                        + f"{out_dict['review/score']},{out_dict['review/time']},{out_dict['review/summary']},{out_dict['review/text']}\n"
                    )
                    out.write(output)
                count += 1
                if count % 1000 == 0:
                    out.flush()
                    print(count)
            elif cells[0] == "review/helpfulness":
                if len(cells) > 1:
                    if "/" in cells[1]:
                        hs = cells[1].split("/")
                        out_dict["hnum"] = int(hs[0])
                        out_dict["hden"] = int(hs[1])
                    else:
                        out_dict["hnum"] = int(cells[1])
                        out_dict["hden"] = int(cells[1])
            elif cells[0] == "review/text":
                if len(cells) > 1:
                    out_dict[cells[0]] = (
                        '"'
                        + ":".join(cells[1:])
                        .replace(",", "")
                        .replace("\n", "")
                        .replace("<br />", "")
                        .replace("\\", "")
                        .replace('"', "")
                        .replace("'", "")
                        .strip()
                        + '"'
                    )
            else:
                if len(cells) > 1:
                    out_dict[cells[0]] = (
                        cells[1]
                        .replace(",", "")
                        .replace("\n", "")
                        .replace("<br />", "")
                        .replace("\\", "")
                        .strip()
                    )

    output = (
        f"{out_dict['product/productId']},{out_dict['review/userId']},{out_dict['review/profileName']},{out_dict['hnum']},{out_dict['hden']},"
        + f"{out_dict['review/score']},{out_dict['review/time']},{out_dict['review/summary']},{out_dict['review/text']}\n"
    )
    out.write(output)
print("======= ALL FINISHED ========")
