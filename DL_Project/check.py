import os

# Directory containing the files
directory = 'C:/Users/Dhruv/PycharmProjects/DeepLearning/Aging/'

# Mapping based on your provided list
mapping = {
    1: "ASD1", 2: "ASD2", 3: "ASD3", 4: "ASD4", 5: "ASD5", 6: "ASD6", 7: "ASD7",
    8: "ASD8", 9: "ASD9", 10: "ASD10", 11: "ASD11", 12: "ASD12", 13: "ASD13",
    14: "ASD14", 15: "ASD15", 16: "ASD16", 17: "ASD17", 18: "ASD18", 19: "ASD19",
    20: "ASD20", 21: "ASD21", 22: "ASD22", 23: "P51", 24: "ASD24", 25: "ASD25",
    26: "ASD26", 27: "ASD27", 28: "ASD28", 29: "ASD29", 30: "P1", 31: "P5",
    32: "P6", 33: "P9", 34: "P10", 35: "P12", 36: "P16", 37: "P17", 38: "P18",
    39: "P20", 40: "P24", 41: "P25", 42: "P26", 43: "P29", 44: "P31", 45: "P32",
    46: "P37", 47: "P38", 48: "P41", 49: "P42", 50: "P43", 51: "P44", 52: "P52",
    53: "P53", 54: "P54", 55: "P56", 56: "P60"
}

# Rename files in the directory
for old_name in os.listdir(directory):
    if old_name.endswith(".set"):
        # Extract the number from the file name
        number = int(old_name.split("Abby")[0])
        # Get the new name from the mapping
        new_base_name = mapping.get(number, None)
        if new_base_name:
            new_name = f"{new_base_name}_Resting.set"
            old_path = os.path.join(directory, old_name)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_name} -> {new_name}")


