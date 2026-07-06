def parse_dat_file(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != ""]

    if len(lines) < 11:
        raise ValueError(f"Malformed .dat file (expected >=11 lines): {path}")

    camera_model = lines[0]
    image_ratio = lines[1]

    matrix = []
    for line in lines[2:11]:
        # cells may be separated by spaces, commas, or no separator at all
        if " " in line or "," in line:
            row = [int(tok) for tok in line.replace(",", " ").split()]
        else:
            row = [int(ch) for ch in line if ch.isdigit()]
        if len(row) != 9:
            raise ValueError(f"Row does not have 9 values in {path}: {line!r}")
        matrix.append(row)

    return {
        "camera_model": camera_model,
        "image_ratio": image_ratio,
        "matrix": matrix,   # list of 9 lists of 9 ints, row-major, 0=empty
    }