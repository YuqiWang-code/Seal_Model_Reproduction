import zipfile, sys
p = "sam3.pt"
with open(p, "rb") as f:
    head = f.read(4)
print("magic:", head)
if head[:2] == b"PK":
    with zipfile.ZipFile(p) as z:
        bad = z.testzip()
        names = z.namelist()[:6]
    print("zip entries ok:", bad is None, "sample:", names)
else:
    print("not a zip; likely legacy pickle")
