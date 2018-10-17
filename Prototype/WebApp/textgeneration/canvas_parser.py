"""
Parser for data provided from Business Model Crawler DB
BMCr_DB_export.json
"""
BMC_Fields = ["Key Partners", "Key Activities", "Value Proposition", "Customer Relationships", "Customer Segments",
                "Key Resources", "Channels", "Cost Structure", "Revenue Streams"]
LMC_Fields = ["Problem",
                "Solution",
                "Key Metrics",
                "Unique Value Proposition",
                "Unfair Advantage",
                "Channels",
                "Customer Segments",
                "Cost Structure",
                "Revenue Stream"]

def canvas_parser():
    import json

   

    with open(file='BMCr_DB_export.json', mode="r", encoding="UTF-8") as f:
        data = json.load(f)
        BusinessModelCanvas = []
        LeanModelCanvas = []
        for x in data["rows"]:
            if x["doc"].get("language", "english") == "english":
                if x["doc"]["modelType"] == "BusinessModelCanvas":
                    BusinessModelCanvas.append(x["doc"]["fields"])
                else:
                    LeanModelCanvas.append(x["doc"]["fields"])
    for field in BMC_Fields:
        with open(file="Dir/BMC " + field + ".txt", mode="a", encoding="utf-8") as newF:
            for item in BusinessModelCanvas:
                for subitem in item:
                    if subitem["name"] == field:
                        line = ["[" + note["title"] + "]: " + note.get("note", "No description") + "." for note in
                                subitem["cards"]]
                        newF.writelines("\n".join(line))
            newF.close()

    for field in LMC_Fields:
        with open(file="Dir/LMC " + field + ".txt", mode="w", encoding="utf-8") as newF:
            for item in LeanModelCanvas:
                for subitem in item:
                    if subitem["name"] == field:
                        line = ["[" + note["title"] + "]: " + note.get("note", "No description") + "." for note in
                                subitem["cards"]]
            newF.writelines("\n".join(line))
            newF.close()

# pprint(data["rows"][0])
