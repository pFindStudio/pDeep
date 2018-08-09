lines = open("modification.ini").readlines()
n = int(lines[0][lines[0].rfind('=')+1:])
mod_py = ["def get_modification():\n    mod_dict = {} # {mod_name: elements}"]
for i in range(n):
    print(lines[(i+1)*2])
    items = lines[(i+1)*2].strip().split("=")
    mod_name = items[0]
    elements = items[1].split(" ")[-1]
    mod_py.append("    mod_dict['{}'] = '{}'".format(mod_name, items[1]))
mod_py.append("    return mod_dict")
with open("modification.py","w") as f: f.write("\n".join(mod_py))
    
    